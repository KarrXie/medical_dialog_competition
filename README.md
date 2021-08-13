# 《蕴含实体的中文医疗对话生成》 比赛 第五名代码

## 说明
代码采用了bert4keras模型框架，并对原码有部分修改。后续会新增pytorch版模型
tensorflow==1.14、 keras==2.3.1

## 代码运行
数据处理：data_process.ipynb
entity预测模型训练：response_entity.py
回复话术生成模型训练：response_generation.py
模型预训练代码和开源的预训练模型并没有差异，因此未添加


## 模型逻辑
### entity预测逻辑
以下列例子为例：
    [{'id': 'Patient', 'Sentence': '胃部不适，第一天有痛感，后面就是胀，不拉肚子。就是胀气。请问是什么原因', 'Symptom': ['腹泻', '腹胀', '胃肠不适'], 'Medicine': [], 'Test': [], 'Attribute': [], 'Disease': []},
    {'id': 'Doctor', 'Sentence': '您好，您的症状有多久了呢？', 'Symptom': [], 'Medicine': [], 'Test': [], 'Attribute': ['时长'], 'Disease': []},
    {'id': 'Doctor', 'Sentence': '平时，有没有反酸嗳气，大便情况怎么样？', 'Symptom': ['打嗝', '反流'], 'Medicine': [], 'Test': [], 'Attribute': [], 'Disease': []}]
这段对话对应的entity预测逻辑为：
    ![Image text](https://github.com/KarrXie/medical_dialog_competition/blob/main/images/entity_predict.png)
 1. 模型采用roformer模型，主要原因是可以处理长文本数据，且效果优于NEZHA
 2. 第一句话中当患者对病情进行描述后，后两句都是医生的response，所以，给定第一句话后，预测医生response中的entity时，需要包含后两句中所有的entity。
 而给定第一句和第二句后，预测的entity应只包括第三句中的entity
 3. 虽然医生的response entity包含在了两个句子中，但是考虑句子先后，所以，在给定第一句话术后，预测entity时，限定P(E2|D1) > P(E3|D1),
 其中，D1表示给定第一句患者说的话，E2 和 E3 分别表示第二句话和第三句话中所含有的entity
 4. 多标签分类算法，采用 softmax+交叉熵。参考：https://spaces.ac.cn/archives/7359  同时，由于负样本占比太大，对其采用欠采样

 ### response generation预测逻辑
   ![Image text](https://github.com/KarrXie/medical_dialog_competition/blob/main/images/response_generation.png)
 1. response generation 采用word_roformer+UniLM模型，本来想尝试gpt模型，但时间原因，没来得及，后续会补上
 2. 一般而已，文本生成采用word-level效果会更好一些。参考：https://spaces.ac.cn/archives/7758