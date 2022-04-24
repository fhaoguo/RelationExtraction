# Relation Extraction v2

v2子项目源于Schlampig/OpenNRE_for_Chinese

## Requirements
  * Python>=3.5
  * pytorch>=0.3.1
  * scikit-learn>=0.18
  * numpy
  * jieba
  * tqdm
  * Flask(optional, if runing the server.py)
<br>

## 数据集
* **source**: The dataset used for this work is from [BaiDu2019 Relation Extraction Competition](http://lic2019.ccf.org.cn/kg), denoted as DuIE. Note that, rather than directly brought into OpenNRE_for_Chinese, DuIE should be first transformed to dataset DuNRE that has the correct format for the model. <br>
* **format of DuIE**: a sample in DuIE is like: <br>
```
sample = {"postag": [{"word": str, "pos": str}, {"word": str, "pos": str}, ...], 
          "text": str,
          "spo_list": [{"predicate": str, "object_type": str, "subject_type": str, "object": str, "subject": str}, 
                       {"predicate": str, "object_type": str, "subject_type": str, "object": str, "subject": str}, 
                       ...]}
```
* **format of DuNRE**: DuNRE contains three main datasets as follows (factually the same format as [OpenNRE](https://github.com/thunlp/OpenNRE)):
```
1. Sample dataset:
    [
        {
            'sentence': str (with space between word and punctuation),
            'head': {'word': str},
            'tail': {'word': str},
            'relation': str (the name of a class)
        },
        ...
    ]
2. Embeddings dataset:
    [
        {'word': str, 'vec': list of float},
        ...
    ]
            
3. Labels dataset:
    {
        'NA': 0 (it is necessary to denote NA as index 0),
        class_name_1 (str): 1,
        class_name_1 (str): 2,
        ...
    }
```
* **example**: a sample would be like:
```
    [
        {
            'sentence': '《 软件 体 的 生命周期 》 是 美国作家 特德·姜 的 作品 ， 2015 年 5 月 译林 出版社 出版 。 译者 张博然 等 。 ',
            'head': {'word': '特德·姜'},
            'tail': {'word': '软件体的生命周期'},
            'relation': ‘作者’)
        },
        ...
    ]
```

<br>


## Command Line:
* **generate DuNRE**: detail steps could be seen [here](prepare_data).
* **prepare**: transform DuNRE to numpy/pickle file for the model.
```bash
python prepare
```
* **train**: train, validate and save the model.
```bash
python learn
```
* **predict**: run the flask server, and predict new samples via Postman.
```bash
python predict
```

<br>
  
## 参考
+ https://github.com/Schlampig/OpenNRE_for_Chinese