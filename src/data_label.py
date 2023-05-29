# _*_ coding: utf-8 _*_
# @Time: 2022/3/24 18:36 
# @Author: 韩春龙

"""
数据集抽取标签生成

"""
import torch
from compare_mt.rouge.rouge_scorer import RougeScorer
import json
import re
from pytorch_transformers import BertTokenizer
from nltk.tokenize import WordPunctTokenizer, WhitespaceTokenizer
from tqdm import tqdm
rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1, rouge2, rougeL = 0, 0, 0
cnt = 0

a = torch.load('../data/数据集保存/cnndm.train.11.bert.pt')
print(a)


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

tokenizer = BertTokenizer.from_pretrained('../temp/bert/bert-base-vocab.txt')
data_ = []
num = 0

# with open('../data/ACL2020_other_datasets/test_wikihow.jsonl', 'r') as cnn:
with open('../data/1/test_multinews.jsonl', 'r') as cnn:
    for line in tqdm(cnn):
        data = json.loads(line)

        source = []
        for txt in data['text']:
            source.append(WordPunctTokenizer().tokenize(txt))

        tgt = []
        for txt in data['summary']:
            tgt.append(WordPunctTokenizer().tokenize(txt))

        sent_labels = greedy_selection(source, tgt, 2)
        src_sent_labels = [0 for i in range(len(data['text']))]
        for j in sent_labels:
            src_sent_labels[j] = 1

        tgt_txt = '<q>'.join(data['summary'])
        sent_txt = data['text']

        sent_input = []
        for txt in data['text']:
            s = []
            content = tokenizer.tokenize(txt)
            s.append(tokenizer.cls_token_id)
            s.extend(tokenizer.convert_tokens_to_ids(content))
            s.append(tokenizer.sep_token_id)
            sent_input.append(s)

        dic = {
            'sent_input': sent_input,
            'src_sent_labels': src_sent_labels,
            'sent_txt': sent_txt,
            'tgt': tgt_txt
        }

        data_.append(dic)

        if len(data_) == 2000:
            torch.save(data_, '../data/xsum/xsum_test_' + str(num) + '.bert.pt')
            num += 1
            data_ = []

    if len(data_) != 0:
        torch.save(data_, '../data/xsum/xsum_test_' + str(num) + '.bert.pt')
