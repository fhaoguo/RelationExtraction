# RelationExtraction v1

v1子项目源于buppt/ChineseNRE，利用关系分类方法，选用bilstm+attention模型，使用词向量+位置向量特征，进行关系抽取。

## Requirements
+ python 3.9
+ pytorch 1.11.0

## 数据
+ buppt/ChineseNRE项目中的people-relation数据
+ 数据格式为: 实体1 实体2 关系 句子

## 训练
### 1.将训练数据处理成pkl文件供模型使用(因有pkl文件，可不执行此步)
运行 `preprocess.py`处理`data/people_relation/train.txt`，得到train.pkl和test.pkl。

### 2.训练模型
运行`python train.py`，可以在`train.py`文件中设置epoch、batch等参数，模型储存到result/v1中。

也可以运行`python train.py pretrained`使用预训练的词向量进行训练(vec.txt是一个训练好的词向量)。


## 参考
+ https://github.com/buppt/ChineseNRE