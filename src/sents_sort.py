# _*_ coding: utf-8 _*_
# @Time: 2022/3/21 15:18
# @Author: 韩春龙   log.txt 2500，768,42.903000,  log1.txt 4500,768 , 42.923000，  log2.txt 256,正常测80阶段 42.960000

from __future__ import division
from datasets import load_metric
import nltk
from operator import itemgetter
from torch.autograd import Variable
import argparse
from torch.nn import functional as F
from tqdm.auto import tqdm
import os
import torch.nn as nn
from torch.nn import LSTMCell
import torch
from transformers import BertModel, BertConfig, BertTokenizer, RobertaModel, RobertaConfig, RobertaTokenizer
from others.logging import logger, init_logger
import random
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import glob
import math
from compare_mt.rouge.rouge_scorer import RougeScorer
import gc
import json
from torch.nn import Parameter
from others.utils import test_rouge, rouge_results_to_str

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        if (args.large):
            self.config = BertConfig.from_pretrained('../temp/bert/large/bert-large-config.json')
            self.model = BertModel.from_pretrained('../temp/bert/large/bert-large-pytorch_model.bin',
                                                   config=self.config)
        else:
            # self.config = BertConfig.from_pretrained('../temp/sentence-transformers_bert-base-nli-mean-tokens/config.json')
            # self.model = BertModel.from_pretrained('../temp/sentence-transformers_bert-base-nli-mean-tokens/pytorch_model.bin', config=self.config)
            self.config = BertConfig.from_pretrained('../temp/bert/bert-base-config.json')
            self.model = BertModel.from_pretrained('../temp/bert/bert-base-pytorch_model.bin',
                                                   config=self.config)

    def forward(self, input_ids, attention_mask, output_hidden_states):

        if (args.mode == "train"):
            top_vec = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=output_hidden_states)

        else:
            self.eval()
            with torch.no_grad():
                top_vec = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=output_hidden_states)

        return top_vec

class RoBerta(nn.Module):
    def __init__(self, args):
        super(RoBerta, self).__init__()
        if (args.large):
            self.config = RobertaConfig.from_pretrained('../temp/Roberta/roberta-base-en')
            self.model = RobertaModel.from_pretrained('../temp/Roberta/roberta-base-en', config=self.config)
            self.model.trian()
        else:
            self.config = RobertaConfig.from_pretrained('../temp/Roberta/roberta-base-en')
            self.model = RobertaModel.from_pretrained('../temp/Roberta/roberta-base-en', config=self.config)

    def forward(self, input_ids, attention_mask):

        if (args.mode == "train"):
            top_vec = self.model(input_ids=input_ids, attention_mask=attention_mask)

        else:
            self.eval()
            with torch.no_grad():
                top_vec = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return top_vec

class ReduceState(nn.Module):
    def __init__(self, args):
        super(ReduceState, self).__init__()
        self.args = args
        self.reduce_h = nn.Linear(2 * args.hidden_size, args.hidden_size)

        self.reduce_c = nn.Linear(2 * args.hidden_size, args.hidden_size)

    def forward(self, hidden):
        # 对维度进行特征整合
        h, c = hidden  # h, c dim = 2,1,hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, 2 * self.args.hidden_size)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, 2 * self.args.hidden_size)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h, hidden_reduced_c)  # h, c dim = 1 x b x hidden_dim

class MogLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, mog_iterations):
        super(MogLSTM, self).__init__()
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.rnn1 = nn.LSTMCell(input_size, hidden_size)
        self.mog_iterations = mog_iterations
        self.Q = Parameter(torch.Tensor(hidden_size, input_size))
        self.R = Parameter(torch.Tensor(input_size, hidden_size))
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.R)


    def mogrify(self, xt, ht):
        for i in range(1, self.mog_iterations + 1):
            if (i % 2 == 0):
                ht = (2 * torch.sigmoid(xt @ self.R)) * ht
            else:
                xt = (2 * torch.sigmoid(ht @ self.Q)) * xt
        return xt, ht

    def forward(self, input):
        recurrent, f_cx = self.rnn(input[:, 0, :])
        fwd = [recurrent]
        for i in range(1, input.shape[1]):
            recurrent, f_cx = self.rnn(input[:, i, :], (recurrent, f_cx))
            fwd.append(recurrent)
        forward = torch.stack(fwd, dim=0).squeeze(1)

        input_reverse = torch.flip(input, dims=[1])
        recurrent_b, b_cx = self.rnn1(input_reverse[:, 0, :])
        bwd = [recurrent_b]
        for i in range(1, input_reverse.shape[1]):
            recurrent_b, b_cx = self.rnn1(input_reverse[:, i, :], (recurrent_b, b_cx))
            bwd.append(recurrent_b)
        backward = torch.stack(bwd, dim=0).squeeze(1)
        backward_reverse = torch.flip(backward, dims=[0])
        return torch.cat((forward, backward_reverse), -1).unsqueeze(0)

class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, args):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(768, 2048)
        self.w_2 = nn.Linear(2048, 768)
        self.layer_norm = nn.LayerNorm(768, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(args.dropout)
        self.dropout_2 = nn.Dropout(args.dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        w = torch.load('../l_w.pt')
        self.linear1.weight = w
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h)
        return sent_scores

class sort_model(nn.Module):
    def __init__(self, args, device, tokenizer, checkpoint=None):
        super(sort_model, self).__init__()
        self.args = args
        self.encoder = Bert(self.args)
        # self.encoder = RoBerta(self.args)
        self.POOLING = args.POOLING
        self.pad_token_id = tokenizer.pad_token_id

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)


        self.dropout = nn.Dropout(args.dropout)
        self.sigmoid = nn.Sigmoid()
        self.Classifier = Classifier(2 * args.hidden_size)

        self.A_d = nn.Linear(2 * args.hidden_size, 2 * args.hidden_size)

        self.fuse_lstm = MogLSTM(args.input_size, args.hidden_size, args.mog_iterations)
        # self.score_lstm = nn.LSTM(args.input_size, args.hidden_size, 1, batch_first=True, bidirectional=True)
        # self.pff = PositionwiseFeedForward(self.args)
        # self.reduce_state = ReduceState(self.args)

        self.to(device)

    def amplifier(self, sent_input_i, sent_input_j):
        for i in range(1, self.args.k + 1):
            h_i = self.A_d(sent_input_i - sum(sent_input_j) / sent_input_j.size(0)) + sent_input_i
            sent_input_i = h_i

        return sent_input_i

    def forward(self, sent_input, src_sent_labels):

        candidate_num = sent_input.size(0)
        candidate_id = sent_input.view(-1, sent_input.size(-1))
        input_mask = candidate_id != self.pad_token_id
        hidden_states = self.encoder(candidate_id, attention_mask=input_mask, output_hidden_states=True)['hidden_states']

        if self.POOLING == 'first_last_avg':
            output_hidden_state = (hidden_states[-1] + hidden_states[0]).mean(dim=1)
        elif self.POOLING == 'last_avg':
            output_hidden_state = (hidden_states[-1]).mean(dim=1)
        elif self.POOLING == 'last2avg':
            output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
        elif self.POOLING == 'cls':
            output_hidden_state = (hidden_states[-1][:, 0, :])
        else:
            raise Exception("unknown pooling {}".format(self.POOLING))

        # candidate_emb 仅仅知识通过bert将句子进行编码不包含任何上下文信息， 使用cls表示每个句子
        # 通过lstm网络将每个句子的cls进行链接整合，使其包含上下文信息，bert加上双向lstm算是完成句子编码
        candidate_emb = output_hidden_state.view(candidate_num, -1).unsqueeze(0)

        sents_feature = self.fuse_lstm(candidate_emb)[0]
        # sents_feature = self.pff(sents_feature.squeeze(0))
        sent_am_fe = []
        for idx, sent_input_i in enumerate(sents_feature):
            if idx == 0:
                sent_input_j = sents_feature[1:]
                am_fe = self.amplifier(sent_input_i, sent_input_j)
                sent_am_fe.append(am_fe)
            else:
                sent_input_j = torch.cat([sents_feature[:idx], sents_feature[idx+1:]], dim=0)
                am_fe = self.amplifier(sent_input_i, sent_input_j)
                sent_am_fe.append(am_fe)

        sents_feature = torch.stack(sent_am_fe, dim=0)
        sent_scores = self.Classifier(sents_feature)

        # sents_feature, s_t = self.fuse_lstm(candidate_emb)
        # sents_feature = self.pff(sents_feature)

        # sent_scores = torch.tensor([]).cuda()
        # # c_t = Variable(torch.zeros((1, 2 * self.args.hidden_size))).cuda()
        #
        # ht = torch.zeros((1, 1, self.args.hidden_size)).to(sent_scores.device)
        # Ct = torch.zeros((1, 1, self.args.hidden_size)).to(sent_scores.device)

        # s_t = self.reduce_state(s_t)

        # for di in range(sents_feature.size(1)):
        #     xt = sents_feature[:, di, :].unsqueeze(1)  # 1 * 768
        #
        #     # x = self.x_context(torch.cat((c_t, y_t), 1))  # 2*h
        #     for t in range(1, self.args.mog_iterations + 1):
        #         if (t % 2 == 0):
        #             ht = (2 * torch.sigmoid(xt @ self.R)) * ht
        #         else:
        #             xt = (2 * torch.sigmoid(ht @ self.Q)) * xt
        #
        #     lstm_out, s_t = self.moglstm(xt, (ht, Ct))
        #
        #     out_put = self.sigmoid(self.out(lstm_out)).squeeze(-1)
        #     sent_scores = torch.cat((sent_scores, out_put.squeeze(-1)), dim=0)
        #
        #     ht, Ct = s_t
        #     # c_t = c_t + x * out_put

        return sent_scores

class Batch(object):
    def _pad(self, data, tokenizer, max_len, labels=False):

        width = max_len
        rtn_data = []
        if labels:
            for d in data:
                if len(d) < width:
                    rtn_data.append(d + [-1] * (width - len(d)))
                else:
                    rtn_data.append(d[:width])
        else:
            for d in data:
                if len(d) > width:
                    rtn_data.append(d[:width - 1] + [tokenizer.sep_token_id])
                else:
                    rtn_data.append(d + [tokenizer.pad_token_id] * (width - len(d)))

        return rtn_data

    def __init__(self, data=None, device=None, is_test=False, tokenizer=None):

        self.batch_size = 1

        if data is not None:

            if is_test:
                sent_input = data[0][0][:80]
                tgt_txt = data[0][1]
                sent_txt = data[0][2][:80]
                src_sent_labels = data[0][3][:80]

                l = max([len(s) for s in sent_input])
                max_sent_len = l if l < 40 else 40
                sent_input = torch.tensor(self._pad(sent_input, tokenizer, max_sent_len))
                src_sent_labels = torch.tensor(src_sent_labels)

                setattr(self, 'sent_input', sent_input.to(device))
                setattr(self, 'tgt_txt', tgt_txt)
                setattr(self, 'sent_txt', sent_txt)
                setattr(self, 'src_sent_labels', src_sent_labels.to(device))

            else:
                sent_input = data[0][0][:80]
                src_sent_labels = data[0][1][:80]

                l = max([len(s) for s in sent_input])
                max_sent_len = l if l < 40 else 40
                sent_input = torch.tensor(self._pad(sent_input, tokenizer, max_sent_len))
                src_sent_labels = torch.tensor(src_sent_labels)

                setattr(self, 'sent_input', sent_input.to(device))
                setattr(self, 'src_sent_labels', src_sent_labels.to(device))

    def __len__(self):
        return self.batch_size


class Dataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test, tokenizer):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets, self.tokenizer)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter, self.tokenizer)

    def _next_dataset_iterator(self, dataset_iter, tokenizer):
        try:
            # 删除不必要的数据减少内存
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args=self.args, dataset=self.cur_dataset, batch_size=self.batch_size, device=self.device,
                            shuffle=self.shuffle, is_test=self.is_test, tokenizer=tokenizer)


class DataIterator(object):
    def __init__(self, args, dataset, batch_size, device=None, is_test=False, shuffle=True, tokenizer=None):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.device = device
        self.shuffle = shuffle
        self.tokenizer = tokenizer

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):  # 6

        if is_test:
            sent_input = ex["sent_input"]
            tgt = ex['tgt']
            sent_txt = ex['sent_txt']
            src_sent_labels = ex['src_sent_labels']
            return sent_input, tgt, sent_txt, src_sent_labels
        else:
            sent_input = ex["sent_input"]
            src_sent_labels = ex['src_sent_labels']

            return sent_input, src_sent_labels

    def batch_buffer(self, data, batch_size, is_test):
        batch = []
        for ex in data:
            ex = self.preprocess(ex, is_test)
            # 没标签
            if not is_test:
                if sum(ex[1]) == 0:
                    continue
            if len(ex[0]) > 200 or len(ex[0]) < 3:
                continue
            batch.append(ex)
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def create_batches(self, batch_size, is_test):
        data = self.data()
        for buffer in self.batch_buffer(data, batch_size, is_test):
            yield buffer

    def __iter__(self):
        while True:
            self.batches = self.create_batches(self.batch_size, self.is_test)
            for idx, minibatch in enumerate(self.batches):
                batch = Batch(minibatch, self.device, self.is_test, self.tokenizer)
                yield batch
            return


def load_dataset(args, corpus_type, shuffle):
    assert corpus_type in ["train", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    pts = sorted(glob.glob(args.data_path + '/cnndm.' + corpus_type + '*.pt'))
    # pts = sorted(glob.glob(args.data_path + '/wiki_' + corpus_type + '*.pt'))

    if pts:
        if (shuffle):
            random.shuffle(pts)
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:

        pt = args.data_path + '/cnndm.train.0.bert.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def doc_sent_loss(score, src_sent_labels):
    # 计算余弦相似度, loss
    loss_func = nn.BCELoss()

    loss = loss_func(score.squeeze(0), src_sent_labels.float())

    return loss


def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _block_tri(c, p):
    tri_c = _get_ngrams(3, c.split())
    for s in p:
        tri_s = _get_ngrams(3, s.split())
        if len(tri_c.intersection(tri_s)) > 0:
            return True
    return False


def test_abs(model, args, tokenizer):
    def test_iter_fct():
        return Dataloader(args, load_dataset(args, 'test', shuffle=False), 1, device,
                          shuffle=False, is_test=True, tokenizer=tokenizer)

    test_iter = test_iter_fct()
    model.eval()
    # metric = load_metric("./rouge.py")

    # rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    # rouge1, rouge2, rougeLsum = 0, 0, 0
    # cnt = 0

    can = []
    god = []
    with torch.no_grad():
        for i, batch in enumerate(test_iter):

            sent_input = batch.sent_input
            tgt_txt = batch.tgt_txt
            sent_txt = batch.sent_txt
            src_sent_labels = batch.src_sent_labels

            score = model(sent_input, src_sent_labels)  # sent_emb, src_emb
            similarity = score.squeeze(0).cpu().tolist()

            index = list(map(itemgetter(0), sorted(enumerate(similarity), key=itemgetter(1), reverse=True)))
            _pred = []
            for idx in index:

                candidate = sent_txt[idx].strip()

                if (not _block_tri(candidate, _pred)):
                    _pred.append(candidate)

                if (len(_pred) == 3):
                    break

            can.append('<q>'.join(_pred))
            god.append(tgt_txt)
            # score = rouge_scorer.score(tgt_txt.replace('<q>', '\n'), '\n'.join(_pred))

            # rouge1 += score["rouge1"].fmeasure
            # rouge2 += score["rouge2"].fmeasure
            # rougeLsum += score["rougeLsum"].fmeasure
            # cnt += 1

    # result = metric.compute(predictions=can, references=god)
    # rouge1 = result["rouge1"].mid.fmeasure
    # rouge2 = result["rouge2"].mid.fmeasure
    # rougeL = result["rougeL"].mid.fmeasure
    #
    temp_dir = '../temp/bert'

    rouges = test_rouge(temp_dir, can, god)

    rouge1 = rouges["rouge_1_f_score"] * 100.0,
    rouge2 = rouges["rouge_2_f_score"] * 100.0,
    rougeL = rouges["rouge_l_f_score"] * 100.0,

    # rouge1 = rouge1 / cnt
    # rouge2 = rouge2 / cnt
    # rougeLsum = rougeLsum / cnt

    model.train()

    logger.info("rouge1: %.6f, rouge2: %.6f, rougeL: %.6f" % (rouge1[0], rouge2[0], rougeL[0]))
    return (rouge1[0] + rouge2[0] + rougeL[0]) / 3


def train_abs(args, device_id):
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    # 随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained('../temp/bert/bert-base-vocab.txt')

    # tokenizer = RobertaTokenizer.from_pretrained('../temp/Roberta/roberta-base-en', cache_dir=None)

    def train_iter_fct():
        return Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device, shuffle=True,
                          is_test=False, tokenizer=tokenizer)

    train_iter = train_iter_fct()

    # 模型，优化器
    model = sort_model(args, device, tokenizer)
    model.to(device)
    init_lr = args.lr / args.warmup_steps
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr)

    # loss_function = Cosine_SoftLoss(device)

    max_r1 = 0
    all_step_cnt = 0
    model.train()
    # 开始训练
    for epoch in range(args.epoch):
        optimizer.zero_grad()
        step_cnt = 0
        sim_step = 0
        avg_loss = 0

        for i, batch in enumerate(train_iter):
            sent_input = batch.sent_input
            src_sent_labels = batch.src_sent_labels

            score = model(sent_input, src_sent_labels)

            loss = doc_sent_loss(score, src_sent_labels) / args.accumulate_step
            avg_loss += loss.item()
            loss.backward()

            step_cnt += 1
            if step_cnt == args.accumulate_step:

                step_cnt = 0
                sim_step += 1
                all_step_cnt += 1

                lr = args.lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.step()
                optimizer.zero_grad()

            if sim_step % args.report_freq == 0 and step_cnt == 0:
                logger.info("epoch: %d, all_step_cnt: %d, avg loss: %.12f" % (
                    epoch + 1, all_step_cnt, avg_loss / (args.accumulate_step * args.report_freq)))
                avg_loss = 0

            if all_step_cnt % 1000 == 0 and all_step_cnt != 0 and step_cnt == 0:
                r1 = test_abs(model, args, tokenizer)
                if r1 > max_r1:
                    max_r1 = r1
                    torch.save(model.state_dict(), os.path.join(args.model_save, str(all_step_cnt) + '.bin'))
                    logger.info("best - epoch: %d, all_step_cnt: %d" % (epoch, all_step_cnt))
                    logger.info("acc : %.6f" % (max_r1))

        r1 = test_abs(model, args, tokenizer)
        if r1 > max_r1:
            max_r1 = r1
            torch.save(model.state_dict(), os.path.join(args.model_save, str(all_step_cnt) + '.bin'))
            logger.info("best - epoch: %d, all_step_cnt: %d" % (epoch, all_step_cnt))
            logger.info("acc : %.6f" % (max_r1))

        train_iter = train_iter_fct()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', choices=['train', 'test'], type=str)
    parser.add_argument("-data_path", default='../data/cnndm/', type=str)
    # parser.add_argument("-data_path", default='../bert_data/cnndm/sents_sort_bert', type=str)
    # parser.add_argument("-data_path", default='../bert_data/wiki', type=str)
    parser.add_argument("-result_path", default='.. /results/', type=str)
    parser.add_argument("-log_file", default='../logs/train4.txt', type=str)
    # parser.add_argument("-model_save", default='../models/wiki/', type=str)
    parser.add_argument("-model_save", default='../models/sort_bert/', type=str)
    parser.add_argument("-test_model_path", default='../models/sort/12600.bin', type=str)

    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-large", default=False, type=bool)
    parser.add_argument("-lr", default=2e-3, type=float)
    parser.add_argument("-warmup_steps", default=2500, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    parser.add_argument("-epoch", default=1, type=int)
    parser.add_argument("-accumulate_step", default=12, type=int)  # 12
    parser.add_argument("-report_freq", default=50, type=int)

    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-seed', default=970903, type=int)

    parser.add_argument('-POOLING', default='cls', type=str)  # last_avg  last2avg  first_last_avg

    parser.add_argument('-dropout', default=0.2, type=float)
    parser.add_argument('-input_size', default=768, type=int)
    parser.add_argument('-hidden_size', default=256, type=int)
    parser.add_argument('-mog_iterations', default=0, type=int)
    parser.add_argument('-k', default=1, type=int)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.mode == 'train'):
        train_abs(args, device_id)
    else:
        tokenizer = BertTokenizer.from_pretrained('../temp/bert/bert-base-vocab.txt', cache_dir=None)
        checkpoint = torch.load(args.test_model_path)
        model = sort_model(args, device, tokenizer)
        model.load_state_dict(checkpoint, strict=True)
        model.to(device)
        test_abs(model, args, tokenizer)
