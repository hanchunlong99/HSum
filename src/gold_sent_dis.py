# _*_ coding: utf-8 _*_
# @Time: 2022/7/6 17:02 
# @Author: 韩春龙

import torch
import glob
from others.utils import test_rouge, rouge_results_to_str
from pytorch_transformers import BertTokenizer
import re
from nltk.tokenize import WordPunctTokenizer, WhitespaceTokenizer
from compare_mt.rouge.rouge_scorer import RougeScorer
import pyecharts.options as opts
from pyecharts.charts import Line
from numpy import mean
from tqdm import tqdm

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

# pts = sorted(glob.glob('../data/cnndm/cnndm.test.*.pt'))
pts = sorted(glob.glob('../data/Multi-news/multi_test_*.pt'))
# pts = sorted(glob.glob('../data/pubmed/pubmed_test*.pt'))
# pts = sorted(glob.glob('../data/wiki/wiki_test*.pt'))
tokenizer = BertTokenizer.from_pretrained('../temp/bert/bert-base-vocab.txt')
rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
temp_dir = '../temp/bert'

data_ = []
pp = []
ababab = []
for pt in pts:
    a = torch.load(pt)
    for data in tqdm(a):
        pz = 0
        for fgdf in data['sent_input']:
            pz += (len(fgdf) - 2)
        ababab.append(pz)
        src_sent_labels = data['src_sent_labels']  # 截断前的标签位置
        tgt_txt = data['tgt']
        sent_txt = data['sent_txt']
        if len(src_sent_labels) == 1 or sum(src_sent_labels) == 0:
            continue

        src_txt_123 = []
        for idx, fg in enumerate(src_sent_labels):
            if fg == 1:
                src_txt_123.append(sent_txt[idx])

        sec = '<q>'.join(src_txt_123)   # 未截断的标摘


        sent_len = 0
        idx = 0

        while sent_len < 510 and idx < len(sent_txt):
            l = len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_txt[idx])))
            sent_len += l
            idx += 1

        sent_txt = sent_txt[:idx]
        if idx == len(sent_txt):
            idx -= 1

        if True:
            pp.append(idx / (len(src_sent_labels)-1))  # 截断位置
            continue

        source = []
        for txt in sent_txt:
            source.append(WordPunctTokenizer().tokenize(txt))

        tgt_12 = []
        tgt_12.append(WordPunctTokenizer().tokenize(tgt_txt))

        sent_labels = greedy_selection(source, tgt_12, 4)   # 截断后的标签位置

        dsfs = []
        for idx in sent_labels:
            dsfs.append(sent_txt[idx])

        src = '<q>'.join(dsfs)   # 截断后标准摘要
        tgt = tgt_txt  # 标准摘要

        # rouges = test_rouge(temp_dir, [sec], [tgt])
        score1 = rouge_scorer.score(tgt, sec)  # 截断前分数
        r1 = score1["rouge1"].fmeasure
        r2 = score1["rouge2"].fmeasure
        rl = score1["rougeL"].fmeasure
        # rouge1 = rouges["rouge_1_f_score"] * 100.0,
        # rouge2 = rouges["rouge_2_f_score"] * 100.0,
        # rougeL = rouges["rouge_l_f_score"] * 100.0,
        score1 = (r1 + r2 + rl)/3

        score2 = rouge_scorer.score(tgt, src)  # 截断后分数
        r1 = score2["rouge1"].fmeasure
        r2 = score2["rouge2"].fmeasure
        rl = score2["rougeL"].fmeasure

        # rouges = test_rouge(temp_dir, [src], [tgt])
        # rouge1 = rouges["rouge_1_f_score"] * 100.0,
        # rouge2 = rouges["rouge_2_f_score"] * 100.0,
        # rougeL = rouges["rouge_l_f_score"] * 100.0,
        score2 = (r1 + r2 + rl)/3

        s = score1 - score2  # 截断损失
        zz = []
        for idz, lp in enumerate(src_sent_labels):
            if lp == 1:
              zz.append(idz)

        g_a1 = sum([x/(len(src_sent_labels)-1) for x in zz])/len(zz)  # 截断前位置均值
        g_a2 = sum([x/(len(src_sent_labels)-1) for x in sent_labels])/len(zz) # 截断后位置均值

        v = {
            'sunshi': s,
            'weizhi1': g_a1,
            'weizhi2': g_a2
        }
        data_.append(v)

mean_data = mean(pp)
print(mean_data)
print(mean(ababab))

sg = sorted(data_, key=lambda x: x['weizhi1'], reverse=False)  # 按照损失进行升序

x_data = []
y_data = []

for dd in sg:
    if dd['sunshi'] <= 0:
        continue
    elif dd['sunshi'] not in x_data:
        y_data.append(dd['sunshi'])
        x_data.append(dd['weizhi1'])

x1, x2, x3, x4, x5 = [], [], [], [], []
y1, y2, y3, y4, y5 = [], [], [], [], []
for idf, xx in enumerate(x_data):
    if xx <= 0.2:
        x1.append(xx)
        y1.append(y_data[idf])
    elif xx <= 0.4:
        x2.append(xx)
        y2.append(y_data[idf])
    elif xx <= 0.6:
        x3.append(xx)
        y3.append(y_data[idf])
    elif xx <= 0.8:
        x4.append(xx)
        y4.append(y_data[idf])
    else:
        x5.append(xx)
        y5.append(y_data[idf])

x_ = []
y_ = []

x_.append('20%')
y_.append(mean(y1))

x_.append('40%')
y_.append(mean(y2))

x_.append('60%')
y_.append(mean(y3))

x_.append('80%')
y_.append(mean(y4))

x_.append('100%')
y_.append(mean(y5))




(
    Line()
    .set_global_opts(
        title_opts=opts.TitleOpts(title="WikiHow"),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            name='Gold sentence location interval',
            name_location='center',
            name_gap=40,
            name_textstyle_opts=opts.TextStyleOpts(font_size=22),
            axislabel_opts=opts.LabelOpts(font_size=22)
        ),

        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
            name='Truncation loss',
            name_location='center',
            name_gap=65,
            name_textstyle_opts=opts.TextStyleOpts(font_size=22),
            axislabel_opts=opts.LabelOpts(font_size=22)
        ),
    )
    .add_xaxis(
        xaxis_data=x_)
    .add_yaxis(
        series_name="",
        y_axis=y_,
        symbol="emptyCircle",
        is_symbol_show=True,
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .render("WikiHow.html")
)






