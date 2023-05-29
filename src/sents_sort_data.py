import torch
import itertools
import glob
from tqdm import tqdm
from pytorch_transformers import BertTokenizer, RobertaTokenizer
import re
import copy
import nltk

from operator import itemgetter

def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences):

    assert len(sentences) > 0
    assert n > 0

    words = sum(sentences, [])

    return _get_ngrams(n, words)

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def greedy_selection(doc_sent_list, abstract_sent_list):

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    abstract = abstract_sent_list
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(s).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    rouge_score = []

    for i in range(len(sents)):
        c = [i]
        candidates_1 = [evaluated_1grams[idx] for idx in c]
        candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[idx] for idx in c]
        candidates_2 = set.union(*map(set, candidates_2))
        rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
        rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
        rouge_score.append(rouge_1 + rouge_2)

    return rouge_score

# 保存版本
pts = sorted(glob.glob('../data/cnndm/cnndm.train.' + '*.pt'))  # 区分超长数据
tokenizer = BertTokenizer.from_pretrained('../temp/bert/bert-base-vocab.txt')
data_new_ = []
for pt in pts:
    a = torch.load(pt)
    data_new = []
    for data in tqdm(a):
        src_txt = copy.deepcopy(data["sent_txt"])
        tgt_txt = copy.deepcopy(data["tgt"])

        sels = greedy_selection(src_txt, [tgt_txt])
        src = []
        src.append(tokenizer.cls_token_id)
        for txt in src_txt:
            content = tokenizer.tokenize(txt)
            src.extend(tokenizer.convert_tokens_to_ids(content))
        src.append(tokenizer.sep_token_id)
        src_len = len(src)

        if src_len > 512:
            continue
            # # 1是核心+辅助，0是干扰
            # classification_index = []
            # classification_labels = [1 for x in range(len(src_txt))]
            # sels = greedy_selection(src_txt, [tgt_txt])
            #
            # while src_len > 512:
            #     min_data = min(sels)
            #     min_index = sels.index(min_data)
            #
            #     classification_index.append(data["src_txt"].index(src_txt[min_index]))
            #     sent_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(src_txt[min_index]))
            #     src_len -= len(sent_token)
            #     del sels[min_index]
            #     del src_txt[min_index]
            #     assert len(sels) == len(src_txt)
            #
            # while min(sels) == 0.0:
            #     min_index = sels.index(min(sels))
            #     classification_index.append(data["src_txt"].index(src_txt[min_index]))
            #
            #     sent_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(src_txt[min_index]))
            #     src_len -= len(sent_token)
            #     del sels[min_index]
            #     del src_txt[min_index]
            #     assert len(sels) == len(src_txt)
            #
            # for i in classification_index:
            #     classification_labels[i] = 0
            #
            # assert len(classification_labels) == len(data["src_txt"]) and src_len <= 512 and min(sels) != 0.0
            #
            # sent_input = []
            #
            # for txt in data['src_txt']:
            #     sent = []
            #     sent.append(tokenizer.cls_token_id)
            #     content = tokenizer.tokenize(txt)
            #     sent.extend(tokenizer.convert_tokens_to_ids(content))
            #     sent.append(tokenizer.sep_token_id)
            #     sent_input.append(sent)
            #
            # dic = {
            #     'src_input': src,
            #     'src_txt': ' '.join(data['src_txt']),
            #     'sent_input': sent_input,
            #     'sent_txt': data['src_txt'],
            #     'classification_labels': classification_labels,
            #     'tgt_input': data['tgt'],
            #     'tgt_txt': data['tgt_txt'],
            #     'src_sent_labels': data['src_sent_labels']
            # }
            #
            # data_new.append(dic)
        else:

            sent_input = []

            for txt in data['src_txt']:
                sent = []
                sent.append(tokenizer.cls_token_id)
                content = tokenizer.tokenize(txt)
                sent.extend(tokenizer.convert_tokens_to_ids(content))
                sent.append(tokenizer.sep_token_id)
                sent_input.append(sent)

            dic = {
                'src_input': src,
                'src_txt': ' '.join(data['src_txt']),
                'sent_input': sent_input,
                'sent_txt': data['src_txt'],
                'classification_labels': '',
                'tgt_input': data['tgt'],
                'tgt_txt': data['tgt_txt'],
                'src_sent_labels': data['src_sent_labels']
            }

            data_new_.append(dic)

    # torch.save(data_new, pt)
torch.save(data_new_, '../data/cnndm/sents_sort/cnndm.train.144.bert.pt')

# pts = sorted(glob.glob('../data/cnndm/cnndm.' + '*.pt'))
# # tokenizer = BertTokenizer.from_pretrained('../temp/bert/bert-base-vocab.txt')
# tokenizer = RobertaTokenizer.from_pretrained('../temp/Roberta/roberta-base-en', cache_dir=None)
#
# for pt in pts:
#     a = torch.load(pt)
#     data_new = []
#     for data in tqdm(a):
#         if len(data['src_txt']) != len(data['src_sent_labels']):
#             continue
#         src_txt = copy.deepcopy(data["src_txt"])
#         tgt_txt = copy.deepcopy(data["tgt_txt"])
#
#         sent_input = []
#         for txt in src_txt:
#             s = []
#             content = tokenizer.tokenize(txt)
#             s.append(tokenizer.cls_token_id)
#             s.extend(tokenizer.convert_tokens_to_ids(content))
#             s.append(tokenizer.sep_token_id)
#             sent_input.append(s)
#
#         dic = {
#             'sent_input': sent_input,
#             'src_sent_labels': data['src_sent_labels'],
#             'sent_txt': data['src_txt'],
#             'tgt': data['tgt_txt']
#         }
#
#         data_new.append(dic)
#
#     print(len(data_new))
#     torch.save(data_new, pt)
