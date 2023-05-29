# _*_ coding: utf-8 _*_
# @Time: 2021/12/11 20:30 
# @Author: 韩春龙
from summary import summarizer
import glob
import torch
from tqdm import tqdm
import re
import copy
from operator import itemgetter
from sklearn.metrics import accuracy_score, f1_score
"""
textrank测准确度
"""
def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

# import math
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



def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1



pts = sorted(glob.glob('../../data/数据集保存/cnndm.test.1' + '*.pt'))  # test

can_path = './cnndm.candidate'
gold_path = './scnndm.gold'
temp_dir = './temp'

# with open(can_path, 'w', encoding='utf-8') as save_pred:
#     with open(gold_path, 'w', encoding='utf-8') as save_gold:
#
#         for pt in pts:
#             a = torch.load(pt)
#             for data in tqdm(a):
#
#                 text = data["src_txt"]
#
#                 text = " ".join(txt if txt[-1] == '.' else txt+' .' for txt in text)
#                 t, _ = summarizer.summarize(text, scores=True)
#                 tgt = []
#                 for i in t:
#                     tgt.append(i[0][:-2])
#                 tgt_ = '<q>'.join(tgt)
#
#                 save_pred.write(tgt_.strip() + '\n')
#                 save_gold.write(data["tgt_txt"].strip() + '\n')
#
#
# rouges = test_rouge(temp_dir, can_path, gold_path)
# logger.info('Rouges \n%s' % (rouge_results_to_str(rouges)))

pd = []
pds = []

for pt in pts:
    a = torch.load(pt)
    data_new = []
    for data in tqdm(a):
        pdd = []
        text = copy.deepcopy(data["src_txt"])
        pds.append(len(text))
        tgt_txt = data["tgt_txt"]
        can_txt = data["src_sent_labels"]
        fgsdf = []
        for ggfg, can in enumerate(can_txt):
            if can == 1:
                fgsdf.append(data["src_txt"][ggfg])
        # sels = greedy_selection(text, [tgt_txt])
        #
        # index_ = list(map(itemgetter(0), sorted(enumerate(sels), key=itemgetter(1), reverse=True)))[:20]

        t = summarizer.summarize(text, scores=True)
        assert len(t) == len(text)
        dfdf = []
        for df in t:
            dfdf.append(df[1])
        index = list(map(itemgetter(0), sorted(enumerate(dfdf), key=itemgetter(1), reverse=True)))[:30]

        # for df in t:
        #     pdd.append(df[1])  # 句子评分

        # index = list(map(itemgetter(0), sorted(enumerate(pdd), key=itemgetter(1), reverse=True)))[:30]  # 最大的5个下标

    #     """删句子"""
    #     index.sort()
    #     data["src_txt"] = [data["src_txt"][i] for i in index]
    #     data_new.append(data)
    #     assert len(data["src_txt"]) <= 10
    #
    # torch.save(data_new, pt)
        """测准确"""

        tgt = []
        for i in index:
            tgt.append(data["src_txt"][i])
        fdg = False
        for df in fgsdf:
            if df not in tgt:
                pd.append(int(0))
                fdg = True
                break
        if not fdg:
            pd.append(int(1))

        # if theme1 in tgt or theme2 in tgt or theme3 in tgt:
        #     pd.append(int(1))
        # else:
        #     pd.append(int(0))

print(sum(pd) / len(pd))
print(sum(pds) / len(pds))