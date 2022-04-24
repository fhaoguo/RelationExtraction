# -*- coding: utf-8 -*-
# @FileName   :preprocess.py
# @Time       :2022/4/24 09:24
# @Author     :fenghaoguo

import os
import codecs
import pickle
import numpy as np
import pandas as pd
from collections import deque
from collections.abc import Iterable

REL_COUNT_THETA = 1500  # 每种关系数量的阈值，用于均衡各关系训练语料的数量
MAX_LEN = 50


def flatten(x):
    """
    以字符串为最小分割单位，进行扁平化处理
    :param x: 如：["junk", ["nested stuff"], [['aa a'], 'bb b'], [[]]]
    :return: []，则返回：['junk', 'nested stuff', 'aa a', 'bb b']
    """
    result = []
    for el in x:
        if isinstance(x, Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)

    return result


def get_dict_rel2id(rel2id=None):
    """
    建立映射关系到ID的映射
    :param rel2id:
    :return:
    """
    with codecs.open('../data/people-relation/relation2id.txt', 'r', 'utf-8') as input_data:
        for line in input_data.readlines():
            line_split = line.split()
            rel2id[line_split[0]] = int(line_split[1])


def process_train_corpus(datas=None, positionE1=None, positionE2=None, labels=None, count=None):
    """
    预处理训练语料
    :param datas:
    :param positionE1:
    :param positionE2:
    :param labels:
    :param count:
    :param total_data:
    :return:
    """
    total = 0
    with codecs.open('../data/people-relation/train.txt', 'r', 'utf-8') as tfc:
        for line in tfc:
            line_split = line.split()
            if count[relation2id[line_split[2]]] < REL_COUNT_THETA:
                sentence = []
                index1 = line_split[3].index(line_split[0])
                position1 = []
                index2 = line_split[3].index(line_split[1])
                position2 = []

                for i, word in enumerate(line_split[3]):
                    sentence.append(word)
                    position1.append(i - index1)
                    position2.append(i - index2)

                datas.append(sentence)
                labels.append(relation2id[line_split[2]])
                positionE1.append(position1)
                positionE2.append(position2)
            count[relation2id[line_split[2]]] += 1
            total += 1

    return total


def X_padding(words):
    """
    把words转为id，并自动补全MAX_LEN长度。
    :param words:
    :return:
    """
    ids = []
    for i in words:
        if i in word2id:
            ids.append(word2id[i])
        else:
            ids.append(word2id["UNKNOWN"])
    if len(ids) >= MAX_LEN:
        return ids[:MAX_LEN]
    ids.extend([word2id["BLANK"]]*(MAX_LEN-len(ids)))

    return ids


def pos(num):
    """
    位置向右平移40
    :param num:
    :return:
    """
    if num < -40:
        return 0
    if num >= -40 and num <= 40:
        return num + 40
    if num > 40:
        return 80


def position_padding(positions):
    """
    位置平移并自动补全MAX_LEN长度
    :param positions:
    :return:
    """
    pos_padding = [pos(i) for i in positions]
    if len(pos_padding) >= MAX_LEN:
        return pos_padding[:MAX_LEN]
    pos_padding.extend([81]*(MAX_LEN-len(pos_padding)))
    return pos_padding


def get_train_data(datas=None, positionE1=None, positionE2=None, labels=None):
    df_data = pd.DataFrame({'words': datas, 'tags': labels,'positionE1':positionE1,'positionE2':positionE2}, index=range(len(datas)))
    df_data['words'] = df_data['words'].apply(X_padding)
    # df_data['tags'] = df_data['tags']
    df_data['positionE1'] = df_data['positionE1'].apply(position_padding)
    df_data['positionE2'] = df_data['positionE2'].apply(position_padding)

    datas = np.asarray(list(df_data['words'].values))
    labels = np.asarray(list(df_data['tags'].values))
    positionE1 = np.asarray(list(df_data['positionE1'].values))
    positionE2 = np.asarray(list(df_data['positionE2'].values))

    print(f'---------- train data info ----------')
    print(f'datas.shape:{datas.shape}')
    print(f'labels.shape:{labels.shape}')
    print(f'positionE1.shape:{positionE1.shape}')
    print(f'positionE2.shape:{positionE2.shape}')

    with open('train.pkl', 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(relation2id, outp)
        pickle.dump(datas, outp)
        pickle.dump(labels, outp)
        pickle.dump(positionE1, outp)
        pickle.dump(positionE2, outp)
    print('** train data finished and saved.')


def process_test_corpus(datas=None, positionE1=None, positionE2=None, labels=None, count=None):
    with codecs.open('../data/people-relation/train.txt', 'r', 'utf-8') as tfc:
        for lines in tfc:
            line = lines.split()
            if count[relation2id[line[2]]] > 1500 and count[relation2id[line[2]]] <= 1800:
                sentence = []
                index1 = line[3].index(line[0])
                position1 = []
                index2 = line[3].index(line[1])
                position2 = []

                for i, word in enumerate(line[3]):
                    sentence.append(word)
                    position1.append(i - 3 - index1)
                    position2.append(i - 3 - index2)
                    i += 1
                datas.append(sentence)
                labels.append(relation2id[line[2]])
                positionE1.append(position1)
                positionE2.append(position2)
            count[relation2id[line[2]]] += 1


def get_test_data(datas, positionE1, positionE2, labels):
    df_data = pd.DataFrame({'words': datas, 'tags': labels, 'positionE1': positionE1, 'positionE2': positionE2},
                           index=range(len(datas)))
    df_data['words'] = df_data['words'].apply(X_padding)
    # df_data['tags'] = df_data['tags']
    df_data['positionE1'] = df_data['positionE1'].apply(position_padding)
    df_data['positionE2'] = df_data['positionE2'].apply(position_padding)

    datas = np.asarray(list(df_data['words'].values))
    labels = np.asarray(list(df_data['tags'].values))
    positionE1 = np.asarray(list(df_data['positionE1'].values))
    positionE2 = np.asarray(list(df_data['positionE2'].values))

    print(f'---------- test data info ----------')
    print(f'datas.shape:{datas.shape}')
    print(f'labels.shape:{labels.shape}')
    print(f'positionE1.shape:{positionE1.shape}')
    print(f'positionE2.shape:{positionE2.shape}')

    with open('test.pkl', 'wb') as outp:
        pickle.dump(datas, outp)
        pickle.dump(labels, outp)
        pickle.dump(positionE1, outp)
        pickle.dump(positionE2, outp)
    print('** test data finished and saved.')


if __name__ == '__main__':
    # print(flatten(["junk", ["nested stuff"], [['aa a'], 'bb b'], [[]]]))
    relation2id = {}
    get_dict_rel2id(relation2id)
    # print(f'relation2id:{relation2id}')

    # 获取训练集
    datas = deque()
    labels = deque()
    positionE1 = deque()
    positionE2 = deque()
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total_data = process_train_corpus(datas, positionE1, positionE2, labels, count)
    # print(total_data, len(datas), count)
    # print(len(positionE1), len(positionE2))

    all_words = flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    # print(sr_allwords)

    set_words = sr_allwords.index
    # print(set_words)
    set_ids = range(1, len(set_words)+1)
    word2id = pd.Series(set_ids, index=set_words)
    # print(word2id)
    id2word = pd.Series(set_words, index=set_ids)
    # print(id2word)

    word2id['BLANK'] = len(word2id) + 1
    word2id['UNKNOWN'] = len(word2id) + 1
    id2word[len(id2word) + 1] = 'BLANK'
    id2word[len(id2word) + 1] = 'UNKNOWN'

    if os.path.exists('train.pkl'):
        with open('train.pkl', 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            relation2id = pickle.load(inp)
            train = pickle.load(inp)
            labels = pickle.load(inp)
            positionE1 = pickle.load(inp)
            positionE2 = pickle.load(inp)
        print('---------- train data info ----------')
        print(f'datas.shape:{train.shape}')
        print(f'labels.shape:{labels.shape}')
        print(f'positionE1.shape:{positionE1.shape}')
        print(f'positionE2.shape:{positionE2.shape}')

    else:
        get_train_data(datas, positionE1, positionE2, labels)

    # 获取测试集
    datas = deque()
    labels = deque()
    positionE1 = deque()
    positionE2 = deque()
    count = [0,0,0,0,0,0,0,0,0,0,0,0]
    if os.path.exists('test.pkl'):
        with open('test.pkl', 'rb') as inp:
            test = pickle.load(inp)
            labels = pickle.load(inp)
            positionE1 = pickle.load(inp)
            positionE2 = pickle.load(inp)
        print('---------- test data info ----------')
        print(f'datas.shape:{test.shape}')
        print(f'labels.shape:{labels.shape}')
        print(f'positionE1.shape:{positionE1.shape}')
        print(f'positionE2.shape:{positionE2.shape}')

    else:
        process_test_corpus(datas, positionE1, positionE2, labels, count)
        get_test_data(datas, positionE1, positionE2, labels)
