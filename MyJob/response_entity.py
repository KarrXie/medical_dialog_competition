import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import json
import re
import numpy as np
import jieba
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='roformer', type=str)
parser.add_argument('--mode', type=str, choices=['train', 'predict'])
parser.add_argument('--epochs', default=15, type=int)
parser.add_argument('--batch_size', default=12, type=int)
parser.add_argument('--total_data_number', default=None, type=int, required=True)
parser.add_argument('--entity_list_path', default="./data/entity_list.txt", type=str)
parser.add_argument('--train_data_path', default="./data/train_total_dialog_for_entity_char.json", type=str)
parser.add_argument('--eval_data_path', default="./data/evalution_data_entity_predict.json", type=str)
parser.add_argument('--test_data_path', default="./data/test_entity_predict_char.json", type=str)
parser.add_argument('--model_dir', default="../NEZHA-Base/", type=str)
parser.add_argument('--fineturn_model_path', default='', type=str)

args = parser.parse_args()

steps_per_epoch = int(args.total_data_number//args.batch_size)
config_path = os.path.join(args.model_dir, "bert_config.json")
checkpoint_path = os.path.join(args.model_dir, "model.ckpt")
dict_path = os.path.join(args.model_dir, "vocab.txt")


def corpus():
    """循环读取语料
    """
    while True:
        with open(args.train_data_path) as f:
            for l in f:
                l = json.loads(l)
                yield l

# 加载并精简词表
# 加载并精简词表, 将无用的token剔除掉
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
)

# 补充词表 + 预测标签数字化
compound_tokens = []
label2num_dict = {}
for l in open(args.entity_list_path, 'r', encoding='utf-8'):
    token = l.strip()
    assert token not in token_dict, "token shouldn't in token_dict"
    label2num_dict[token] = len(label2num_dict)
    token_dict[token] = len(token_dict)
    enclude_char = []
    for _t in token:
        if token_dict.get(_t) is not None:
            enclude_char.append(token_dict.get(_t))
    compound_tokens.append(enclude_char if enclude_char else [0])

# 建立分词器
tokenizer = Tokenizer(token_dict, do_lower_case=True)
num2label_dict = dict([(val, key) for key, val in label2num_dict.items()])


def output_normalize(entity_list):
    output_list = [0] * len(label2num_dict)
    if not entity_list:
        return output_list
    for entity in entity_list:
        output_list[label2num_dict[entity]] = 1
    return output_list


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_num = len(label2num_dict)

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_mask_ids, batch_y_true_ids = [], [], [], []
        batch_order_ids = []
        for is_end, texts_dict in self.sample(random):
            token_ids, segment_ids, mask_ids, y_true_ids = [tokenizer._token_start_id], [texts_dict["begin_role"]], [0], [[0] * self.label_num]
            order_ids = [[100] * self.label_num]
            assert len(texts_dict["input"]) == len(texts_dict["output"]), "input and output should be same length"
            for i, (input_text, output_text) in enumerate(zip(texts_dict["input"], texts_dict["output"])):
                assert len(input_text) == len(output_text), "singal input and output should be same length"
                for j, text in enumerate(input_text):   # 每个角色可能在每轮说多句话
                    _entity_lst = []
                    temp_order = [100] * self.label_num
                    for k, lst in enumerate(output_text[j:]):
                        _entity_lst.extend(lst)
                        for _entity in lst:
                            temp_order[label2num_dict[_entity]] = k
                    ids = tokenizer.encode(text)[0][1:]
                    # ids[0] = tokenizer._token_mask_id
                    # if len(token_ids) + len(ids) <= maxlen:   # 在数据清理时，就已经将数据限制在1024了
                    token_ids.extend(ids)
                    segment_ids.extend([(i + segment_ids[0]) % 2] * len(ids))    # 医生先说话，此处为1
                    # if last_round_doctor_speak_round == 1:   # 当前需要预测的医生话术，为医生此轮话术的第一句
                    if (i + segment_ids[0]) % 2 == 1:
                        if _entity_lst:
                            # print(output_text[j])
                            mask_ids.extend([1] + [2] * (len(ids) - 1))  # mask_ids 等于2表示进行第二个任务的预测
                        else:
                            if np.random.rand() > 0.7:  # 0.9
                                mask_ids.extend([1] + [2] * (len(ids) - 1))
                            else:
                                mask_ids.extend([2] + [0] * (len(ids) - 1))
                    else:
                        mask_ids.extend([2] + [0] * (len(ids) - 1))  # 只是用于attention mask
                    y_true_ids.extend([output_normalize(_entity_lst)] * len(ids))
                    order_ids.extend([temp_order] * len(ids))

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_mask_ids.append(mask_ids)
            batch_y_true_ids.append(y_true_ids)
            batch_order_ids.append(order_ids)
            # print("***", len(batch_token_ids), len(batch_segment_ids), len(batch_mask_ids), len(batch_y_true_ids))
            if len(batch_token_ids) == self.batch_size or is_end:
                maxlen = max(len(token_ids) for token_ids in batch_token_ids)
                batch_token_ids = sequence_padding(batch_token_ids, length=maxlen)
                batch_segment_ids = sequence_padding(batch_segment_ids, length=maxlen)
                batch_mask_ids = sequence_padding(batch_mask_ids, length=maxlen)
                batch_y_true_ids = sequence_padding(batch_y_true_ids, length=maxlen)
                batch_order_ids = sequence_padding(batch_order_ids, length=maxlen)
                yield [batch_token_ids, batch_segment_ids, batch_y_true_ids, batch_order_ids, batch_mask_ids], None
                batch_token_ids, batch_segment_ids, batch_mask_ids, batch_y_true_ids, batch_order_ids = [], [], [], [], []


def eval_data_process(data_path, predict=False):
    batch_token_ids, batch_segment_ids, batch_mask_ids, batch_y_true_ids, batch_texts_dict = [], [], [], [], []
    with open(data_path, "r", encoding="utf8") as f:
        for line in f:
            texts_dict = json.loads(line)
            batch_texts_dict.append(texts_dict)
            token_ids, segment_ids, mask_ids = [tokenizer._token_start_id], [texts_dict["begin_role"]], [0]
            y_true_ids = [[0] * len(label2num_dict)]

            for i, input_text in enumerate(texts_dict["input"]):
                for j, text in enumerate(input_text):  # 每个角色可能在每轮说多句话
                    ids = tokenizer.encode(text)[0][1:]
                    token_ids.extend(ids)
                    segment_ids.extend([(i + segment_ids[0]) % 2] * len(ids))  # 医生先说话，此处为1
                    mask_ids.extend([2] + [0] * (len(ids) - 1))
                    y_true_ids.extend([y_true_ids[0]] * len(ids))
            token_ids.append(0)
            segment_ids.append(0)
            mask_ids.append(1)
            if predict:
                y_true_ids.append(y_true_ids[0])   # 预测时，塞入的y_true_ids用不上
            else:
                y_true_ids.append(output_normalize(texts_dict["output"][-1][-1]))

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_mask_ids.append(mask_ids)
            batch_y_true_ids.append(y_true_ids)

        maxlen = max(len(token_ids) for token_ids in batch_token_ids)
        print("eval data maxlen:", maxlen)
        batch_token_ids = sequence_padding(batch_token_ids, length=maxlen)
        batch_segment_ids = sequence_padding(batch_segment_ids, length=maxlen)
        batch_mask_ids = sequence_padding(batch_mask_ids, length=maxlen)  # 将mask_ids当作attention计算
        batch_y_true_ids = sequence_padding(batch_y_true_ids, length=maxlen)
        return [batch_token_ids, batch_segment_ids, batch_y_true_ids, batch_y_true_ids, batch_mask_ids]


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """

    def compute_loss(self, inputs, mask=None):
        _, _, y_true, y_pos, y_mask, y_pred, _ = inputs  # y_logits shape:(None, len(compound_tokens))
        y_pred = y_pred[:, :-1, :]
        y_true = y_true[:, 1:, :]
        y_pos = y_pos[:, 1:, :]
        y_mask = y_mask[:, 1:]
        # order part
        y_pos = K.cast(y_pos[..., None, :] > y_pos[..., None], K.floatx())
        y_pos_positive = tf.multiply(y_pos, y_true[..., None, :])
        sub_y_pred = y_pred[..., None, :] - y_pred[..., None]
        usful_sub_y = tf.multiply(sub_y_pred, y_pos_positive)
        usful_sub_y = usful_sub_y - (1 - y_pos_positive) * 1e12
        _batch_sz = tf.shape(usful_sub_y)
        usful_sub_y = K.reshape(usful_sub_y, [args.batch_size, -1, usful_sub_y.shape[2] * usful_sub_y.shape[3]])
        zeros = K.zeros_like(y_pred[..., :1])
        usful_sub_y = K.concatenate([usful_sub_y, zeros], axis=-1)
        order_loss = tf.reduce_logsumexp(usful_sub_y, axis=-1)
        self.add_metric(order_loss, name='order_loss', aggregation='mean')
        # classify part
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
        y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
        neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
        pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)
        total_loss = neg_loss + pos_loss + order_loss
        y_mask = K.equal(y_mask, 1)
        y_mask = tf.cast(y_mask, total_loss.dtype)
        # total_loss = tf.reduce_sum(total_loss*y_mask)/tf.reduce_sum(y_mask)
        total_loss = tf.reduce_sum(total_loss * y_mask, axis=-1) / (tf.reduce_sum(y_mask, axis=-1)+K.epsilon())
        self.add_metric(total_loss, name='entity_loss', aggregation='mean')
        return total_loss

def get_special_mask(inp):
    y_pred = inp[2][:, :-1, :]
    y_true = inp[0][:, 1:, :]
    y_mask = inp[1][:, 1:]
    y_mask = K.equal(y_mask, 1)  # mask等于1的部分需要进行实体预测
    y_pred = tf.boolean_mask(y_pred, y_mask)
    y_true = tf.boolean_mask(y_true, y_mask)
    return [y_true, y_pred]

_model = build_transformer_model(
    config_path,
    checkpoint_path=checkpoint_path,
    model=args.model_name,
    application='unilm',
    output_logits=True,
    output_entity_logits=True,
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    compound_tokens=compound_tokens,  # 要扩充的词表
    additional_input_layers=[keras.layers.Input(shape=(None, None,)), keras.layers.Input(shape=(None, None,)), keras.layers.Input(shape=(None,))]
)

if args.fineturn_model_path:
    _model.load_weights(args.fineturn_model_path)
output = CrossEntropy([2, 4, 5])(_model.inputs + _model.outputs)  # 此处，输入有三个Input，输出有两个Output
output = keras.layers.Lambda(lambda x: get_special_mask(x))(output)
model = Model(_model.inputs, output)
model.summary()

AdamW = extend_with_weight_decay(Adam, 'AdamW')
AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
optimizer = AdamWG(
    learning_rate=1e-5,
    weight_decay_rate=0.01,
    exclude_from_weight_decay=['Norm', 'bias'],
    grad_accum_steps=16
)
model.compile(optimizer=optimizer)


class Evaluator(keras.callbacks.Callback):
    """保存模型权重
    """
    def __init__(self):
        super(Evaluator, self).__init__()

    def Recall(self, y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + 1e-7)
        return recall

    def Precision(self, y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + 1e-7)
        return precision

    def F1(self, y_true, y_pred):
        precision = self.Precision(y_true, y_pred)
        recall = self.Recall(y_true, y_pred)
        f1 = 2 * ((precision * recall) / (precision + recall + 1e-7))
        return f1

    def eval_pred(self):
        y_true, y_pred = self.model.predict(self.validation_data, batch_size=8)
        best_thred, best_r, best_p, best_f1 = -100, -100, -100, -100
        for thred in np.arange(-5, 5, 0.05):
            pred_output = np.greater(y_pred, thred)
            pred_output = pred_output.astype(int)
            _recall = self.Recall(y_true, pred_output)
            _precision = self.Precision(y_true, pred_output)
            _f1 = self.F1(y_true, pred_output)
            if _f1 > best_f1:
                best_thred = thred
                best_p = _precision
                best_r = _recall
                best_f1 = _f1
        return best_thred, best_r, best_p, best_f1

    def on_epoch_end(self, epoch, logs=None):
        # model.save_weights('./weights/history_entity/model_loss_{}.h5'.format(logs['loss']))
        _thred, _recall, _precision, _f1 = self.eval_pred()
        model.save_weights('./weights/entity/model_loss_thred{:.3f}_r{:.3f}_p{:.3f}_f{:.3f}.h5'.format(_thred, _recall, _precision, _f1))


def scheduler(epoch):
    lr = 1e-5
    if epoch == 2:
        lr = 2e-5
    elif epoch == 3:
        lr = 3e-5
    elif epoch > 4:
        lr = 2e-5
    elif epoch > 6:
        lr = 1e-5
    return lr


if __name__ == '__main__':
    if args.mode == 'train':
        reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)
        evaluator = Evaluator()
        evaluator.validation_data = eval_data_process(args.eval_data_path)
        evaluator.model = model
        train_generator = data_generator(corpus(), args.batch_size)

        model.fit(
            train_generator.forfit(),
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            callbacks=[evaluator, reduce_lr]
        )
    else:
        validation_data = eval_data_process(args.test_data_path, predict=True)
        _, pred_output = model.predict(validation_data, batch_size=8)
        thred_val = args.fineturn_model_path.split('model_loss_thred')[-1].split('_')[0]
        output_num = np.sum(np.greater(pred_output, thred_val).astype(int), axis=-1)
        pred_output = np.argsort(-pred_output)
        with open(args.test_data_path, "r", encoding="utf8") as f:
            batch_texts_dict = []
            for line in f:
                batch_texts_dict.append(json.loads(line))
        with open("./data/entity_predicted_for_generation_predict.json", "w", encoding="utf8") as f:
            for i, (texts_dict, _output) in enumerate(zip(batch_texts_dict, pred_output.tolist())):
                temp_list = []
                for val in _output[:output_num[i]]:
                    temp_list.append(num2label_dict[val])
                if temp_list:
                    add_part = "【" + "，".join(temp_list) + "】"
                else:
                    add_part = "【】"
                texts_dict["input"].append([add_part])
                f.write(json.dumps(texts_dict, ensure_ascii=False) + "\n")