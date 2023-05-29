# _*_ coding: utf-8 _*_
# @Time: 2022/6/30 11:20 
# @Author: 韩春龙


import torch
import glob
from others.utils import test_rouge, rouge_results_to_str
from pytorch_transformers import BertTokenizer
import re
from nltk.tokenize import WordPunctTokenizer, WhitespaceTokenizer
"""抽取式上限计算"""
# # pts = sorted(glob.glob('../bert_data/cnndm/sents_sort_bert/cnndm.test.*.pt'))
# # pts = sorted(glob.glob('../bert_data/multi_news/multi_test*.pt'))
# pts = sorted(glob.glob('../bert_data/wiki/wiki_test*.pt'))
#
# src = []
# tgt = []
#
# for pt in pts:
#     a = torch.load(pt)
#     for data in a:
#         src_sent_labels = data['src_sent_labels']
#         tgt_txt = data['tgt']
#         sent_txt = data['sent_txt']
#         src_txt = []
#         for idx, fg in enumerate(src_sent_labels):
#             if fg == 1:
#                 src_txt.append(sent_txt[idx])
#
#         src.append('<q>'.join(src_txt))
#         tgt.append(tgt_txt)
#
# temp_dir = '../temp/bert'
#
# rouges = test_rouge(temp_dir, src, tgt)
#
# rouge1 = rouges["rouge_1_f_score"] * 100.0,
# rouge2 = rouges["rouge_2_f_score"] * 100.0,
# rougeL = rouges["rouge_l_f_score"] * 100.0,
# print("rouge1: %.6f, rouge2: %.6f, rougeL: %.6f" % (rouge1[0], rouge2[0], rougeL[0]))

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
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

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)

# pts = sorted(glob.glob('../bert_data/cnndm/sents_sort_bert/cnndm.test.*.pt'))
# pts = sorted(glob.glob('../bert_data/multi_news/multi_test*.pt'))
pts = sorted(glob.glob('../data/pubmed/pubned_test*.pt'))
tokenizer = BertTokenizer.from_pretrained('../temp/bert/bert-base-vocab.txt')

src = []
tgt = []

for pt in pts:
    a = torch.load(pt)
    for data in a:

        tgt_txt = data['tgt']
        sent_txt = data['sent_txt']
        sent_len = 0
        idx = 0
        while sent_len < 510 and idx < len(sent_txt):
            l = len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_txt[idx])))
            sent_len += l
            idx += 1

        sent_txt = sent_txt[:idx]

        source = []
        for txt in sent_txt:
            source.append(WordPunctTokenizer().tokenize(txt))

        tgt_12 = []
        tgt_12.append(WordPunctTokenizer().tokenize(tgt_txt))

        sent_labels = greedy_selection(source, tgt_12, 4)

        dsfs = []
        for idx in sent_labels:
            dsfs.append(sent_txt[idx])

        src.append('<q>'.join(dsfs))
        tgt.append(tgt_txt)

temp_dir = '../temp/bert'

rouges = test_rouge(temp_dir, src, tgt)

rouge1 = rouges["rouge_1_f_score"] * 100.0,
rouge2 = rouges["rouge_2_f_score"] * 100.0,
rougeL = rouges["rouge_l_f_score"] * 100.0,
print("rouge1: %.6f, rouge2: %.6f, rougeL: %.6f" % (rouge1[0], rouge2[0], rougeL[0]))