# _*_ coding: utf-8 _*_
# @Time: 2022/1/25 18:12 
# @Author: 韩春龙

from __future__ import division
from operator import itemgetter
import argparse
import os
import torch.nn as nn
import torch
from transformers import RobertaModel, BertModel, BertConfig, BertTokenizer, RobertaTokenizer
from others.logging import logger, init_logger
import random
import numpy as np
import torch.optim as optim
import glob
from compare_mt.rouge.rouge_scorer import RougeScorer
import gc
"""
生成测试数据集用1
"""

class Roberta(nn.Module):
    def __init__(self, args):
        super(Roberta, self).__init__()
        if(args.large):
            # self.model = RobertaModel.from_pretrained('roberta-large', cache_dir=args.temp_dir)
            self.model = RobertaModel.from_pretrained('../Roberta/roberta-base-en', cache_dir=None)
        else:
            # self.model = RobertaModel.from_pretrained('roberta-base', cache_dir=args.temp_dir)
            self.model = RobertaModel.from_pretrained('../Roberta/roberta-base-en', cache_dir=None)
    def forward(self, input_ids, attention_mask):
        # input_id, token_type, mask, clss
        if(args.mode=="train"):
            # top_vec = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
            top_vec = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return top_vec

class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        if(args.large):
            self.config = BertConfig.from_pretrained('../temp/bert-base-config.json')
            self.model = BertModel.from_pretrained('../temp/bert-base-pytorch_model.bin', config=self.config)
        else:
            self.config = BertConfig.from_pretrained('../temp/bert-base-config.json')
            self.model = BertModel.from_pretrained('../temp/bert-base-pytorch_model.bin', config=self.config)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # input_id, token_type, mask, clss

        if(args.mode=="train"):
            top_vec = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        else:
            self.eval()
            with torch.no_grad():
                top_vec = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        return top_vec

class sort_model(nn.Module):
    def __init__(self, args, device, tokenizer, checkpoint=None):
        super(sort_model, self).__init__()
        if args.model_type == 'bert':
            self.encoder = Bert(args)
        else:
            self.encoder = Roberta(args)
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)

        self.pad_token_id = tokenizer.pad_token_id
        # self.expansion1 = nn.Linear(768, 768, bias=True)
        # self.expansion2 = nn.Linear(768, 768, bias=True)
        # self.expansion3 = nn.Linear(768, 768, bias=True)
        # self.tanh = nn.Tanh()
        # self.expansion1 = nn.Linear(768, 1024, bias=True)
        # self.expansion2 = nn.Linear(768, 1024, bias=True)
        # self.expansion3 = nn.Linear(768, 1024, bias=True)

        self.to(device)

    def forward(self, text_id, candidate_id, summary_id=None, require_gold=True):

        batch_size = text_id.size(0)

        input_mask = text_id != self.pad_token_id
        out = self.encoder(text_id, attention_mask=input_mask)[0]
        doc_emb = out[:, 0, :]  # batch len dim

        if require_gold:
            # get reference score
            input_mask = summary_id != self.pad_token_id
            out = self.encoder(summary_id, attention_mask=input_mask)[0]
            summary_emb = out[:, 0, :]  # batch * 1 * 768
            summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)

        candidate_num = candidate_id.size(1)  # batch num len
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))  # batch num len ->batch*num len
        input_mask = candidate_id != self.pad_token_id
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num, -1)

        # get candidate score
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1)

        output = {'score': score}
        if require_gold:
            output['summary_score'] = summary_score
        return output


class Batch(object):
    def _pad(self, data, tokenizer, max_len):

        width = max_len
        rtn_data = []
        for d in data:
            if len(d) > width:
                rtn_data.append(d[:width-1] + [tokenizer.sep_token_id])
            else:
                rtn_data.append(d + [tokenizer.pad_token_id] * (width - len(d)))

        return rtn_data

    def __init__(self, data=None, device=None, is_test=False, tokenizer=None):
        """
        src_txt, tgt_txt, src_input_ids, token_type_id, clss, adjacency_matrix, probability_matrix, graph, scores
        """
        if data is not None:
            self.batch_size = len(data)

            src_txt = [" ".join(x[0]) for x in data]
            candidate = [x[2] for x in data]

            candidate_ = []
            can_ = []
            src_input = torch.LongTensor(self._pad([tokenizer.encode(txt) for txt in src_txt], tokenizer, 512))

            max_txt_num = max(len(d) for d in candidate)
            for can in candidate:
                for txt in can:
                    can_.append(tokenizer.encode(txt['txt']))
                can_ = self._pad(can_, tokenizer, 50)
                can_.extend([[tokenizer.pad_token_id for x in range(50)]] * (max_txt_num - len(can_)))
                candidate_.append(can_)
                can_ = []

            candidate = torch.LongTensor(candidate_)

            tgt_txt = [x[1] for x in data]
            tgt_txt = torch.LongTensor(self._pad([tokenizer.encode(txt) for txt in tgt_txt], tokenizer, 160))
            setattr(self, 'src_input', src_input.to(device))
            setattr(self, 'candidate', candidate.to(device))
            setattr(self, 'tgt_input', tgt_txt.to(device))

            if is_test:
                setattr(self, 'tgt_txt', [x[1] for x in data])
                setattr(self, 'candidate_txt', [x[2] for x in data])



    def __len__(self):
        return self.batch_size

class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
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

        return DataIterator(args=self.args, dataset=self.cur_dataset,  batch_size=self.batch_size, device=self.device, shuffle=self.shuffle, is_test=self.is_test, tokenizer=tokenizer)

class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, is_test=False, shuffle=True, tokenizer=None):
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

    def preprocess(self, ex):  # 6

        src_txt = ex["src_txt"]
        tgt_txt = ex["tgt_txt"]
        candidate = ex["candidate"]

        return src_txt, tgt_txt, candidate

    def batch_buffer(self, data, batch_size):
        batch = []
        for ex in data:
            ex = self.preprocess(ex)
            batch.append(ex)
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def create_batches(self, batch_size):
        data = self.data()
        for buffer in self.batch_buffer(data, batch_size):
            yield buffer

    def __iter__(self):
        while True:
            self.batches = self.create_batches(self.batch_size)
            for idx, minibatch in enumerate(self.batches):
                batch = Batch(minibatch, self.device, self.is_test, self.tokenizer)
                yield batch
            return

def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    """
    对于包含个样本的batch数据 D(X1,X2,y),X1,X2是给定的待排序的两个输入，y代表真实的标签，
    属于{1,-1,当y是1时，X1排在X2前面，反之X2排在X1前面}。
    第n个样本对应的计算如下:
    Ln = max(0,-y*(x1-x2)+margin)
    若X1,X2排序正确且-y*(x1-x2)+margin>margin,则loss为零其他情况loss为-y*(x1-x2)+margin
    """
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)  # 全零矩阵
    # candidate loss
    n = score.size(1)  # batch * n *768
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]  # 不取最后一个
            neg_score = score[:, i:]  # 不取第一个
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)  # 0.01
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss

def load_dataset(args, corpus_type, shuffle):

    assert corpus_type in ["train", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        print('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset


    pts = sorted(glob.glob(args.data_path + '/cnndm.' + corpus_type +'*.pt'))

    if pts:
        if (shuffle):
            random.shuffle(pts)
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:

        pt = args.data_path + '/cnndm.train.0.bert.pt'
        yield _lazy_dataset_loader(pt, corpus_type)

def test_abs(model, args, tokenizer):
    def test_iter_fct():
        return Dataloader(args, load_dataset(args, 'test', shuffle=False), 1, device,
                                      shuffle=False, is_test=True, tokenizer=tokenizer)
    test_iter = test_iter_fct()

    model.eval()
    cnt = []

    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            src_input = batch.src_input
            candidate = batch.candidate

            output = model(src_input, candidate, require_gold=False)
            similarity = output['score'].squeeze(0).tolist()

            index = list(map(itemgetter(0), sorted(enumerate(similarity), key=itemgetter(1), reverse=True)))
            cnt.append(index)
    torch.save(cnt, '../sort_lab.pt')

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

    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('../temp/bert-base-vocab.txt')
    else:
        tokenizer = RobertaTokenizer.from_pretrained('../Roberta/roberta-base-en', cache_dir=None)

    def train_iter_fct():
        return Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device, shuffle=True, is_test=False, tokenizer=tokenizer)

    train_iter = train_iter_fct()

    # 模型，优化器
    model = sort_model(args, device, tokenizer)
    model.to(device)
    model.train()
    init_lr = args.lr / args.warmup_steps
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    minimum_loss = 10000
    all_step_cnt = 0
    # 开始训练
    for epoch in range(args.epoch):
        optimizer.zero_grad()
        step_cnt = 0
        sim_step = 0
        avg_loss = 0
        for i, batch in enumerate(train_iter):
            src_input = batch.src_input
            tgt_input = batch.tgt_input
            candidate = batch.candidate

            output = model(src_input, candidate, tgt_input)
            similarity, gold_similarity = output['score'], output['summary_score']

            loss = args.scale * RankingLoss(similarity, gold_similarity, args.margin, args.gold_margin,
                                            args.gold_weight)
            loss = loss / args.accumulate_step
            avg_loss += loss.item()
            loss.backward()

            step_cnt += 1
            if step_cnt == args.accumulate_step:
                # optimize step
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                step_cnt = 0
                sim_step += 1
                all_step_cnt += 1
                lr = args.lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()

            if sim_step % args.report_freq == 0 and step_cnt == 0:
                logger.info("epoch: %d, batch: %d, avg loss: %.12f" % (epoch+1, sim_step, avg_loss / args.accumulate_step))
                logger.info(f"learning rate: {lr:.12f}")

                avg_loss = 0

            if all_step_cnt % 100 == 0 and all_step_cnt != 0 and step_cnt == 0:
                loss = test_abs(model, args, tokenizer)
                if loss < minimum_loss:
                    minimum_loss = loss
                    torch.save(model.state_dict(), os.path.join(args.model_save, str(all_step_cnt) + '.bin'))
                    logger.info("best - epoch: %d, all_step_cnt: %d" % (epoch, all_step_cnt))
                    logger.info("val rouge: %.6f" % (1 - loss))

        train_iter = train_iter_fct()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='test', choices=['train', 'test'], type=str)
    parser.add_argument("-model_type", default='roberta', type=str)
    parser.add_argument("-data_path", default='../data/cnndm/服务器上的', type=str)
    parser.add_argument("-result_path", default='../results/', type=str)
    parser.add_argument("-temp_dir", default='../Roberta/', type=str)
    parser.add_argument("-log_file", default='../logs/train.txt', type=str)
    parser.add_argument("-model_save", default='../models/sort', type=str)
    parser.add_argument("-test_model_path", default='../models/sort/6900.bin', type=str)

    parser.add_argument("-batch_size", default=10, type=int)
    parser.add_argument("-scale", default=1, type=float)
    parser.add_argument("-margin", default=0.01, type=float)
    parser.add_argument("-gold_margin", default=0, type=float)
    parser.add_argument("-gold_weight", default=1, type=float)

    parser.add_argument("-large", default=False, type=bool)
    parser.add_argument("-lr", default=2e-3, type=float)
    parser.add_argument("-warmup_steps", default=1000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    parser.add_argument("-damping", default=0.8, type=float)
    parser.add_argument("-epoch", default=5, type=int)
    parser.add_argument("-accumulate_step", default=12, type=int)  # 12
    parser.add_argument("-grad_norm", default=0, type=int)  # 12
    parser.add_argument("-report_freq", default=5, type=int)  # 10

    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-seed', default=42, type=int)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1


    if (args.mode == 'train'):
        train_abs(args, device_id)
    else:
        tokenizer = RobertaTokenizer.from_pretrained('../Roberta/roberta-base-en', cache_dir=None)
        checkpoint = torch.load(args.test_model_path)
        model = sort_model(args, device, tokenizer)
        model.load_state_dict(checkpoint, strict=True)
        model.to(device)
        test_abs(model, args, tokenizer)

