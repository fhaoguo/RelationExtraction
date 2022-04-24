# -*- coding: utf-8 -*-
# @FileName   :train.py
# @Time       :2022/4/24 22:36
# @Author     :fenghaoguo

import os
import torch
import json
from v3.opennre.encoder.bert_encoder import BERTEncoder
from v3.opennre.model.softmax_nn import SoftmaxNN
from v3.opennre.framework.sentence_re import SentenceRE

root_path = '/content/drive/My Drive'  # colab环境
# root_path = './'  # 本地环境

# Check data
rel2id = json.load(open(os.path.join(root_path, 'data/people-relation/people-relation_rel2id.json')))
print(rel2id)

sentence_encoder = BERTEncoder(
    max_length=80,
    pretrain_path=os.path.join(root_path, 'pretrain/chinese_wwm_pytorch'),
    mask_entity='store_true'
)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
model = model.to(device)
ckpt = os.path.join(root_path, 'ckpt/v3/model.pth')

# Define the whole training framework
framework = SentenceRE(
    train_path=os.path.join(root_path, 'data/people-relation/people-relation_train.txt'),
    val_path=os.path.join(root_path, 'data/people-relation/people-relation_val.txt'),
    test_path=os.path.join(root_path, 'data/people-relation/people-relation_val.txt'),
    model=model,
    ckpt=ckpt,
    batch_size=64, # Modify the batch size w.r.t. your device
    max_epoch=1,
    lr=2e-5,
    opt='adamw'
)

# Train the model
framework.train_model()

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('Accuracy on test set: {}'.format(result['acc']))

