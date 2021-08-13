import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
import re
import json
import jieba
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets_v4 import sequence_padding, open
from bert4keras.snippets_v4 import DataGenerator, AutoRegressiveDecoder
#from tensorflow.keras.models import Model
from keras.models import Model
from MyJob.evaluate_new import eval_output
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='roformer', type=str)
parser.add_argument('--mode', type=str, choices=['train', 'predict'])
parser.add_argument('--epochs', default=15, type=int)
parser.add_argument('--learning_rate', default=4e-5, type=float)
parser.add_argument('--batch_size', default=12, type=int)
parser.add_argument('--max_generation_len', default=50, type=int)
parser.add_argument('--entity_list_path', default="./data/entity_list.txt", type=str)
parser.add_argument('--add_whole_word_except_entity_path', default="./data/roformer_add_whole_word_except_entity.txt", type=str)
parser.add_argument('--train_data_path', default="./data/train_total_dialog_for_entity_char.json", type=str)
parser.add_argument('--train_data_path', default="./data/train_total_dialog_for_entity_char.json", type=str)
parser.add_argument('--eval_data_path', default="./data/evalution_data_entity_predict.json", type=str)
parser.add_argument('--test_data_add_entity_path', default="./data/test_data_add_entity.json", type=str)
parser.add_argument('--test_response_path', default="./data/response.txt", type=str)
parser.add_argument('--model_dir', default="../NEZHA-Base/", type=str)
parser.add_argument('--fineturn_model_path', default='', type=str)

args = parser.parse_args()

steps_per_epoch = int(args.total_data_number//args.batch_size)
config_path = os.path.join(args.model_dir, "bert_config.json")
checkpoint_path = os.path.join(args.model_dir, "model.ckpt")
dict_path = os.path.join(args.model_dir, "vocab.txt")

# chinese_roformer_L-12_H-768_A-12是一个word-level的bert模型，它的vocab词库没有包含全当前医疗文本的高频词，需要手动加上。
# 此处新增的词是当前语料采用jieba分词且频次在8次以上的词。对于频次低的词，拆封成char后进行训练

jieba.load_userdict(args.entity_list_path)
high_freq_jieba = [line.strip() for line in open(args.add_whole_word_except_entity_path, "r", encoding="utf8")]

def corpus():
    """循环读取语料
    """
    while True:
        with open(args.train_data_path) as f:
            for l in f:
                l = json.loads(l)
                yield l

# 加载并精简词表
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + high_freq_jieba,
)

# 补充词表 + 预测标签数字化
compound_tokens = []
label2num_dict = {"others": 0}
for l in open('./data/entity_list.txt', 'r', encoding='utf-8'):
    token = l.strip()
    assert token not in token_dict, "token shouldn't in token_dict"
    label2num_dict[token] = len(label2num_dict)
    token_dict[token] = len(token_dict)
    enclude_char = []
    for _t in token:
        if token_dict.get(_t) is not None:
            enclude_char.append(token_dict.get(_t))
    compound_tokens.append(enclude_char if enclude_char else [0])

tokenizer = Tokenizer(token_dict, do_lower_case=True, pre_tokenize=lambda x: jieba.cut(x, HMM=False))


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, _random=False):
        _print = 0
        batch_token_ids, batch_segment_ids, batch_mask_ids, batch_y_true_ids, batch_attention_ids = [], [], [], [], []
        for is_end, texts_dict in self.sample(_random):
            _print += 1
            token_ids, segment_ids, mask_ids = [tokenizer._token_start_id], [texts_dict["begin_role"]], [0]
            charactor_shift_time = len(texts_dict["input"])
            for i, (input_text, entity_list) in enumerate(zip(texts_dict["input"], texts_dict["output"])):
                # if i == charactor_shift_time - 1 and random.random() > 0.3:
                if i == charactor_shift_time - 1:
                    input_text = ["".join(input_text)]
                    temp_list = []
                    for _en in entity_list:
                        temp_list.extend(_en)
                    # random.shuffle(temp_list)
                    entity_list = [temp_list]
                for j, (text, _entity_list) in enumerate(zip(input_text, entity_list)):   # 每个角色可能在每轮说多句话
                    ids = tokenizer.encode(text)[0][1:]
                    if i == charactor_shift_time - 1:
                    # if i == charactor_shift_time - 1 and j == last_round_doctor_speak_round - 1:
                        entity_part = "【" + "，".join(_entity_list) + "】" if _entity_list else "【】"
                        entity_ids = tokenizer.encode(entity_part)[0][1:-1]
                        ids = entity_ids + ids
                        token_ids.extend(ids)
                        segment_ids.extend([(i + segment_ids[0]) % 2] * len(ids))  # 医生先说话，此处为1
                        # mask_ids[-1] = 1

                        mask_ids.extend([0] * len(entity_ids) + [2] * (len(ids) - len(entity_ids)))  # mask_ids 等于2表示进行第二个任务的预测
                    else:
                        token_ids.extend(ids)
                        segment_ids.extend([(i + segment_ids[0]) % 2] * len(ids))  # 医生先说话，此处为1
                        mask_ids.extend([0] * len(ids))
            # y_true_ids = self._output_normalize(texts_dict["output"])
            if _print % 1000 == 0:
                print(texts_dict["input"])
                print(token_ids)
                print(mask_ids)
                print(segment_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_mask_ids.append(mask_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                maxlen = max(len(token_ids) for token_ids in batch_token_ids)
                batch_token_ids = sequence_padding(batch_token_ids, length=maxlen)
                batch_segment_ids = sequence_padding(batch_segment_ids, length=maxlen)
                batch_mask_ids = sequence_padding(batch_mask_ids, length=maxlen)
                # yield [batch_token_ids, batch_segment_ids, batch_mask_ids, batch_y_true_ids], None
                yield [batch_token_ids, batch_segment_ids, batch_mask_ids], None
                batch_token_ids, batch_segment_ids, batch_mask_ids = [], [], []

class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """
    def compute_loss(self, inputs, mask=None):
        loss = self.compute_loss_of_seq2seq(inputs, mask)
        return loss

    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, _, y_mask, y_pred = inputs
        y_mask = K.equal(y_mask, 2)   # mask等于2的部分需要进行seq2seq预测
        y_mask = K.cast(y_mask, K.floatx())[:, 1:]
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


_model = build_transformer_model(
    config_path,
    checkpoint_path=checkpoint_path,
    model='roformer',
    application='unilm',
    # output_logits=True,
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    compound_tokens=compound_tokens,  # 要扩充的词表
    # return_keras_model=False,
    # ignore_invalid_weights=True,
    additional_input_layers=[keras.layers.Input(shape=(None,))]   # 此处的最大文本长度为 1024
)

if args.fineturn_model_path:
    _model.load_weights(args.fineturn_model_path)

output = CrossEntropy([-1])(_model.inputs + _model.outputs)  # 此处，输入有三个Input，输出有两个Output
model = Model(_model.inputs, output)
model.summary()

AdamW = extend_with_weight_decay(Adam, 'AdamW')
AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
optimizer = AdamWG(
    learning_rate=args.learning_rate,
    weight_decay_rate=0.01,
    exclude_from_weight_decay=['Norm', 'bias'],
    grad_accum_steps=16
)
model.compile(optimizer=optimizer)


def scheduler(epoch):
    lr = 4e-5
    if epoch > 5:
        lr = 3e-5
    elif epoch > 7:
        lr = 2e-5
    elif epoch > 9:
        lr = 1e-5
    return lr


reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)


class ChatBot(AutoRegressiveDecoder):
    """基于随机采样对话机器人
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids,states=None):
        token_ids, segment_ids, attention_mask = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        curr_segment_ids = np.ones_like(output_ids)
        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)
        # y_true_ids = np.concatenate([y_true_ids, y_true_ids[:len(output_ids)]], 1)
        attention_mask = np.concatenate([attention_mask, curr_segment_ids], 1)
        seq_output = _model.predict([token_ids, segment_ids, attention_mask])
        return seq_output[:, -1]   # 此处模型输出为mlm的softmax输出

    def response(self, texts_dicts, topk=1):
        token_ids, segment_ids = [tokenizer._token_start_id], [texts_dicts["begin_role"]]
        attention_mask = [0]
        for i, text_list in enumerate(texts_dicts["input"]):
            for text in text_list:
                ids = tokenizer.encode(text)[0][1:]
                token_ids.extend(ids)
                segment_ids.extend([(i + segment_ids[0]) % 2] * len(ids))
                attention_mask.extend([0] * len(ids))
        token_ids.pop()   # 去除尾部的【SEP】
        segment_ids.pop()
        attention_mask.pop()
        results = self.random_sample([token_ids, segment_ids, attention_mask], 1, topk)
        return tokenizer.decode(results[0])


class Evaluator(keras.callbacks.Callback):
    """保存模型权重
    """
    def __init__(self, data_path):
        super(Evaluator, self).__init__()
        self.num_passed_batchs = 0
        self.context = []
        self.reference = []
        self.submit = []
        self.load_eval_data(data_path)

    def load_eval_data(self, data_path):
        with open(data_path, "r", encoding="utf8") as f:
            for line in f:
                line = re.sub("\$|##", "", line)
                single_dt = json.loads(line)
                self.reference.extend(single_dt['input'].pop())
                response_entity = single_dt['output'][-1][-1]
                single_dt['input'].append(["【" + "，".join(response_entity) + "】"])
                self.context.append(single_dt)
        print("evalution data sample:", self.context[0], "***", self.reference[0])

    def on_batch_begin(self, batch, logs=None):
        if self.num_passed_batchs < steps_per_epoch:
            K.set_value(self.model.optimizer.lr, args.learning_rate * (self.num_passed_batchs+1) / steps_per_epoch)
            self.num_passed_batchs += 1

    def on_epoch_end(self, epoch, logs=None):
        chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=50)
        for _context in self.context:
            self.submit.append(chatbot.response(_context))
        ave, f1, bleuave = eval_output(self.submit, self.reference)
        model.save_weights('./weights/generation_v4.2/model_ave_{}_f1_{}_bleeave_{}.h5'.format(ave, f1, bleuave))
        self.submit = []


if __name__ == '__main__':
    if args.mode == 'train':
        evaluator = Evaluator(args.eval_data_path)
        evaluator.model = model
        train_generator = data_generator(corpus(), args.batch_size)

        model.fit(
            train_generator.forfit(),
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            callbacks=[evaluator]
        )
    else:
        chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=args.max_generation_len)
        with open(args.test_response_path, "w", encoding="utf8") as f:
            for line in open(args.test_data_add_entity_path, "r", encoding="utf8"):
                line = re.sub("\$|##", "", line)   # 在entity预测时，会将entity前后分别加上‘##’和‘$’,从而将它们视为单个char，但是在generation时不需要，因为此处本身就采用jieba分词了
                line_dict = json.loads(line.strip())
                response = chatbot.response(line_dict)
                # response = response1 if len(set(response1)) > 1 else response2
                f.write(response + '\n')
