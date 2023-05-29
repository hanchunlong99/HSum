import torch
import torch.nn as nn
from torch.nn import Parameter
from tqdm import tqdm
import json
import nltk
import nltk.data

"""
M-news,数据集处理
"""
data_src = []
data_tgt = []

tokenizer = nltk.data.load('../punkt/english.pickle')

with open('../data/Multi-news/drive-download-20220621T053225Z-001/test.txt.src', 'r', encoding='utf-8') as cnn:
    for line in tqdm(cnn):
        data_src.append(line.replace('   ', '').strip())

with open('../data/Multi-news/drive-download-20220621T053225Z-001/test.txt.tgt', 'r', encoding='utf-8') as cnn:
    for line in tqdm(cnn):
        data_tgt.append(line.replace(line[0], '').strip())


with open("../data/Multi-news/multi_test.jsonl", "w", encoding='utf-8') as f:
    for idx in range(len(data_src)):
        dict_ = {
            'text': tokenizer.tokenize(data_src[idx]),
            'summary': tokenizer.tokenize(data_tgt[idx])
        }
        f.write(json.dumps(dict_) + "\n")



