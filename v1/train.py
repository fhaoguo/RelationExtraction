# -*- coding: utf-8 -*-
# @FileName   :train.py
# @Time       :2022/4/23 23:02
# @Author     :fenghaoguo


import numpy as np
import pickle
import sys
import codecs
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from model import Model

with open('train.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    relation2id = pickle.load(inp)
    train = pickle.load(inp)
    labels = pickle.load(inp)
    position1 = pickle.load(inp)
    position2 = pickle.load(inp)

with open('test.pkl', 'rb') as inp:
    test = pickle.load(inp)
    labels_t = pickle.load(inp)
    position1_t = pickle.load(inp)
    position2_t = pickle.load(inp)

print("train len:{}".format(len(train)))
print("test len:{}".format(len(test)))
print("word2id len:{}".format(len(word2id)))

NUM_WORKERS = 0
EMBEDDING_SIZE = len(word2id) + 1
EMBEDDING_DIM = 100

POS_SIZE = 82  # 不同数据集这里可能会报错。
POS_DIM = 25

HIDDEN_DIM = 200

TAG_SIZE = len(relation2id)

BATCH_SIZE = 128
EPOCHS = 3  # 100

config = {
    'EMBEDDING_SIZE': EMBEDDING_SIZE,
    'EMBEDDING_DIM': EMBEDDING_DIM,
    'POS_SIZE': POS_SIZE,
    'POS_DIM': POS_DIM,
    'HIDDEN_DIM': HIDDEN_DIM,
    'TAG_SIZE': TAG_SIZE,
    'BATCH_SIZE': BATCH_SIZE,
    "pretrained": False}

learning_rate = 0.0005  # 0.0005

embedding_pre = []
if len(sys.argv) == 2 and sys.argv[1] == "pretrained":
    print("use pretrained embedding")
    config["pretrained"] = True
    word2vec = {}
    with codecs.open('vec.txt', 'r', 'utf-8') as input_data:
        for line in input_data.readlines():
            word2vec[line.split()[0]] = map(eval, line.split()[1:])

    unknow_pre = []
    unknow_pre.extend([1] * 100)
    embedding_pre.append(unknow_pre)  # wordvec id 0
    for word in word2id:
        if word in word2vec:
        # if word2vec.has_key(word):  # old version
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unknow_pre)

    embedding_pre = np.asarray(embedding_pre)
    print(f'embedding_pre.shape:{embedding_pre.shape}')

model = Model(config, embedding_pre)
# model = torch.load('model/model_epoch20.pkl')
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(reduction='mean')
# criterion = nn.CrossEntropyLoss(size_average=True)  # old version

train = torch.LongTensor(train[:len(train) - len(train) % BATCH_SIZE])
position1 = torch.LongTensor(position1[:len(train) - len(train) % BATCH_SIZE])
position2 = torch.LongTensor(position2[:len(train) - len(train) % BATCH_SIZE])
labels = torch.LongTensor(labels[:len(train) - len(train) % BATCH_SIZE])
train_datasets = TensorDataset(train, position1, position2, labels)
train_dataloader = DataLoader(train_datasets, BATCH_SIZE, True, num_workers=NUM_WORKERS)

test = torch.LongTensor(test[:len(test) - len(test) % BATCH_SIZE])
position1_t = torch.LongTensor(position1_t[:len(test) - len(test) % BATCH_SIZE])
position2_t = torch.LongTensor(position2_t[:len(test) - len(test) % BATCH_SIZE])
labels_t = torch.LongTensor(labels_t[:len(test) - len(test) % BATCH_SIZE])
test_datasets = TensorDataset(test, position1_t, position2_t, labels_t)
test_dataloader = DataLoader(test_datasets, BATCH_SIZE, True, num_workers=NUM_WORKERS)

loss_lst = []
for epoch in range(EPOCHS):
    print(f"---------- epoch:{epoch+1} ----------")
    acc = 0
    total = 0

    loss_bat = 0.
    for sentence, pos1, pos2, tag in train_dataloader:
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence, pos1, pos2)
        tags = Variable(tag)
        loss = criterion(y, tags)
        loss_bat += loss.detach().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y = np.argmax(y.data.numpy(), axis=1)

        for y1, y2 in zip(y, tag):
            if y1 == y2:
                acc += 1
            total += 1

    loss_lst.append(loss_bat / BATCH_SIZE)
    loss_bat = 0.
    print("train:{:.2f}%".format(100 * float(acc) / total))

    acc_t = 0
    total_t = 0
    count_predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    count_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    count_right = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for sentence, pos1, pos2, tag in test_dataloader:
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence, pos1, pos2)
        y = np.argmax(y.data.numpy(), axis=1)
        for y1, y2 in zip(y, tag):
            count_predict[y1] += 1
            count_total[y2] += 1
            if y1 == y2:
                count_right[y1] += 1

    precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(count_predict)):
        if count_predict[i] != 0:
            precision[i] = float(count_right[i]) / count_predict[i]

        if count_total[i] != 0:
            recall[i] = float(count_right[i]) / count_total[i]

    precision = 100 * sum(precision) / len(relation2id)
    recall = 100 * sum(recall) / len(relation2id)
    print("准确率：{:.2f}%".format(precision))
    print("召回率：{:.2f}%".format(recall))
    print("f：{:.2f}%".format((2 * precision * recall) / (precision + recall)))

    if epoch > 0 and epoch % 20 == 0:
        model_name = "../ckpt/v1/epoch" + str(epoch) + ".pkl"
        torch.save(model, model_name)
        print(model_name, " has been saved")

torch.save(model, "../ckpt/v1/model.pkl")
print("model has been saved")

import matplotlib.pyplot as plt
x = [i for i in range(1, len(loss_lst)+1)]
plt.plot(x, loss_lst)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.title('Loss curve')
plt.show()


if __name__ == '__main__':
    run_code = 0
