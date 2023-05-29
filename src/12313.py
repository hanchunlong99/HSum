# _*_ coding: utf-8 _*_
# @Time: 2022/4/1 20:49 
# @Author: 韩春龙
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

# dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
#
# model = SentenceTransformer('bert-base-nli-mean-tokens', cache_folder='./')
#
# train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=1),
#                   InputExample(texts=['Another pair', 'Unrelated sentence'], label=0)]
#
# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1)
# train_loss = losses.SoftmaxLoss(model)
#
# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

"""
得到的sent_embedding是经过池化和全联接的
"""
#
# a = torch.load('../data/cnndm/test6/cnndm.test.6.bert.pt')
# print(a)
import json
import glob
import copy
from pytorch_transformers import BertTokenizer
from tqdm import tqdm

# pts = sorted(glob.glob('../data/cnndm/sents_sort/cnndm.train.' + '*.pt'))
# tokenizer = BertTokenizer.from_pretrained('../temp/bert/bert-base-vocab.txt')

# sels = []
# with open('../data/cnndm/labels.json', "r", encoding='utf-8') as fw:
#     for num, d in enumerate(fw):
#         sels.append(json.loads(d)['labels'])  # 句子排序
# num123 = 0
# for pt in pts:
#     a = torch.load(pt)
#     data_new = []
#     for data in tqdm(a):
#         if len(data['sent_txt']) == len(data['src_sent_labels']):
#             sel = sels[num123]
#             assert len(sel) == len(data['sent_txt'])
#
#             src = []
#             for num, txt in enumerate(data['sent_txt']):
#                 content = tokenizer.tokenize(txt)
#                 src.append(tokenizer.cls_token_id)
#                 src.extend(tokenizer.convert_tokens_to_ids(content))
#                 src.append(tokenizer.sep_token_id)
#
#             l = len(src)
#             src_txt = copy.deepcopy(data['sent_txt'])
#
#             while (l > 512):
#                 min_data = min(sel)
#                 min_index = sel.index(min_data)
#
#                 data_len = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(src_txt[min_index]))
#                 l -= (len(data_len) + 2)
#                 del sel[min_index]
#                 del src_txt[min_index]
#                 assert len(sel) == len(src_txt)
#
#             segs_new = []
#             src_new = []
#
#             for num, txt in enumerate(src_txt):
#                 content = tokenizer.tokenize(txt)
#
#                 src_new.append(tokenizer.cls_token_id)
#                 src_new.extend(tokenizer.convert_tokens_to_ids(content))
#                 src_new.append(tokenizer.sep_token_id)
#
#                 if (num + 1) % 2 == 0:
#                     segs_new.extend([1 for x in range(len(content) + 2)])
#                 else:
#                     segs_new.extend([0 for x in range(len(content) + 2)])
#
#             assert len(src_new) == len(segs_new) == l and len(src_new) <= 512
#
#             data['src_input'] = src_new
#             data['segs'] = segs_new
#             data_new.append(data)
#             num123 += 1
#
#     torch.save(data_new, pt)


pts = sorted(glob.glob('../data/数据集保存/cnndm.test.0' + '*.pt'))
tokenizer = BertTokenizer.from_pretrained('../temp/bert/bert-base-vocab.txt')

datas = []
for pt in pts:
    a = torch.load(pt)
    for data in a:
        if len(data['sent_txt']) == len(data['src_sent_labels']):
            datas.append(data)
        else:
            continue

data_new = []


for data in datas:
    src_txt = data['sent_txt']
    segs_new = []
    src_new = []
    for num, txt in enumerate(src_txt):
        content = tokenizer.tokenize(txt)

        src_new.append(tokenizer.cls_token_id)
        src_new.extend(tokenizer.convert_tokens_to_ids(content))
        src_new.append(tokenizer.sep_token_id)

        if (num + 1) % 2 == 0:
            segs_new.extend([1 for x in range(len(content) + 2)])
        else:
            segs_new.extend([0 for x in range(len(content) + 2)])

    data['src_input'] = src_new
    data['segs'] = segs_new

    data_new.append(data)

torch.save(data_new, '../data/cnndm/cnndm.train.144.bert.pt')
