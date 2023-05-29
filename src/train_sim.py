# _*_ coding: utf-8 _*_
# @Time: 2022/3/14 9:06
# @Author: 韩春龙

# 14 新备份
from __future__ import division
from operator import itemgetter
import argparse
from torch.nn import functional as F
import os
import torch.nn as nn
import torch
from transformers import BertModel, BertConfig, BertTokenizer
from others.logging import logger, init_logger
import random
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import glob
from compare_mt.rouge.rouge_scorer import RougeScorer
import gc
import itertools
import math

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        super(MultiHeadedAttention, self).__init__()
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)
        key = shape(key)
        value = shape(value)
        query = shape(query)

        query = query / math.sqrt(dim_per_head)  # batch * 12 * seq * 64
        scores = torch.matmul(query, key.transpose(2, 3))  # batch * 12 * 64 * seq

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)

        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context

class DecoderLayer(nn.Module):
    def __init__(self, dec_hidden_size, dec_heads, dec_ff_size, dec_dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            dec_heads, dec_hidden_size, dropout=dec_dropout)

        self.context_attn = MultiHeadedAttention(
            dec_heads, dec_hidden_size, dropout=dec_dropout)

        self.feed_forward = PositionwiseFeedForward(dec_hidden_size, dec_ff_size, dec_dropout)
        self.layer_norm_1 = nn.LayerNorm(dec_hidden_size, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(dec_hidden_size, eps=1e-6)
        self.drop = nn.Dropout(dec_dropout)

    def forward(self, inputs, memory_bank, encoder_pad_mask):
        input_norm = self.layer_norm_1(inputs)  # cls

        query = self.self_attn(input_norm, input_norm, input_norm).squeeze(1)

        query = self.drop(query) + inputs  # 没有seq_len
        query_norm = self.layer_norm_2(query)
        mid = self.context_attn(memory_bank, memory_bank, query_norm,
                                mask=encoder_pad_mask).squeeze(1)  # 消掉seq_len
        output = self.feed_forward(self.drop(mid) + query)

        return output

class Decoder(nn.Module):
    def __init__(self, dec_layers, dec_hidden_size, dec_heads, dec_ff_size, dec_dropout):
        super(Decoder, self).__init__()
        self.num_layers = dec_layers
        self.transformer_layers = nn.ModuleList(
            [DecoderLayer(dec_hidden_size, dec_heads, dec_ff_size, dec_dropout)
             for _ in range(dec_layers)])
        self.layer_norm = nn.LayerNorm(dec_hidden_size, eps=1e-6)

    def forward(self, txt_id, cls_embedding, encoder_embedding, padding_idx):
        txt_id = txt_id[:, 1:]
        encoder_batch, encoder_len = txt_id.size()  # 1 413
        output = cls_embedding
        # cls_num = output.size(0)

        encoder_pad_mask = txt_id.data.eq(padding_idx).unsqueeze(1) \
            .expand(encoder_batch, 1, encoder_len)


        for i in range(self.num_layers):
            output = self.transformer_layers[i](
                output, encoder_embedding,
                encoder_pad_mask)

        output = self.layer_norm(output)
        return output

class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        if(args.large):
            self.config = BertConfig.from_pretrained('../temp/bert-base-config.json')
            # self.config.attention_probs_dropout_prob = 0.3
            # self.config.hidden_dropout_prob = 0.3
            self.model = BertModel.from_pretrained('../temp/bert-base-pytorch_model.bin', config=self.config)

        else:
            self.config = BertConfig.from_pretrained('../temp/bert-base-config.json')
            # self.config.attention_probs_dropout_prob = 0.3
            # self.config.hidden_dropout_prob = 0.3
            self.model = BertModel.from_pretrained('../temp/bert-base-pytorch_model.bin', config=self.config)
            # for name, param in self.model.named_parameters():
            #     param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        if(args.mode=="train"):
            top_vec = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

        else:
            self.eval()
            with torch.no_grad():
                top_vec = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

        return top_vec

class sort_model(nn.Module):
    def __init__(self, args, device, tokenizer, checkpoint=None):
        super(sort_model, self).__init__()
        self.args = args
        self.encoder = Bert(self.args)
        self.decoder = Decoder(self.args.doc_dec_layers, self.args.doc_dec_hidden_size, self.args.doc_dec_heads,
                               self.args.doc_dec_ff_size, self.args.doc_dec_dropout)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)

        self.pad_token_id = tokenizer.pad_token_id
        self.to(device)

    def forward(self, src_id, sents_id):

        input_mask = src_id != self.pad_token_id
        output_doc = self.encoder(src_id, attention_mask=input_mask)
        doc_emb = self.decoder(src_id, output_doc[:, 0, :], output_doc[:, 1:, :], 0)

        input_mask = sents_id != self.pad_token_id
        output_sum = self.encoder(sents_id, attention_mask=input_mask)
        summary_emb = self.decoder(sents_id, output_sum[:, 0, :], output_sum[:, 1:, :], 0)

        output = {
            'doc_emb': doc_emb,
            'summary_emb': summary_emb
        }

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
        src_txt, tgt_txt, src_sent_labels, sent_txt
        """
        if data is not None:
            self.batch_size = len(data)

            src_txt = [x[0] for x in data]
            tgt_txt = [x[1] for x in data]
            src_sent_labels = [x[2] for x in data][0]
            sents = [x[3] for x in data][0]

            l = max([len(s.split(' ')) for s in src_txt])
            max_src_len = l if l < 512 else 512
            src_input = torch.LongTensor(self._pad([tokenizer.encode(txt) for txt in src_txt], tokenizer, max_src_len))

            l = max([len(s.split(' ')) for s in sents])
            max_sent_len = l if l < 60 else 60
            sent_input = torch.LongTensor(self._pad([tokenizer.encode(txt) for txt in sents], tokenizer, max_sent_len))

            src_sent_labels = torch.tensor(src_sent_labels)

            setattr(self, 'src_input', src_input.to(device))
            setattr(self, 'sent_input', sent_input.to(device))
            setattr(self, 'src_sent_labels', src_sent_labels.to(device))

            if is_test:
                tgt = data[0][1]  # 正常摘要
                sent_txt = data[0][3]  # 正常的句子

                can_ = []

                for sent in sent_txt:
                    can_.append(tokenizer.encode(sent))

                l = max([len(s) for s in can_])
                max_sent_len = l if l < 60 else 60
                can_ = self._pad(can_, tokenizer, max_sent_len)  # 一个batch中的padding
                candidate = torch.LongTensor(can_)
                setattr(self, 'candidate', candidate.to(device))
                setattr(self, 'sent_txt', sent_txt)
                setattr(self, 'tgt', tgt)


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

    def preprocess(self, ex, is_test):  # 6

        src_txt = ex["src"]
        tgt_txt = ex["tgt_txt"]
        sent_txt = ex["sent_txt"]
        src_sent_labels = ex["src_sent_labels"]
        return src_txt, tgt_txt, src_sent_labels, sent_txt


    def batch_buffer(self, data, batch_size, is_test):
        batch = []
        for ex in data:
            ex = self.preprocess(ex, is_test)
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

def doc_summ_loss(doc_emb, summary_emb, src_sent_labels):
    # 计算余弦相似度, loss
    doc_emb = doc_emb.expand_as(summary_emb)
    summary_score = F.cosine_similarity(doc_emb.unsqueeze(1), summary_emb.unsqueeze(0), dim=2)
    summary_similarities = summary_score / args.lamda
    loss_f = torch.nn.BCELoss()
    sigmoid = nn.Sigmoid()
    loss = loss_f(sigmoid(summary_similarities[0]), src_sent_labels.float())
    return loss

def load_dataset(args, corpus_type, shuffle):

    assert corpus_type in ["train", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
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
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    model.eval()
    cnt = 0

    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            src_input = batch.src_input
            sent3 = batch.candidate
            sent_txt = batch.sent_txt
            tgt = batch.tgt
            # src_input = src_input.expand(sent3.size(0), -1)
            output = model(src_input, sent3)
            doc, summary = output['doc_emb'], output['summary_emb']

            doc = doc.expand_as(summary)

            summary_score = F.cosine_similarity(doc.unsqueeze(1), summary.unsqueeze(0), dim=2) / args.lamda
            pre_score = summary_score[0]

            index = list(map(itemgetter(0), sorted(enumerate(pre_score), key=itemgetter(1), reverse=True)))[:3]
            sents = "<q>".join(sent_txt[i] for i in index)

            score = rouge_scorer.score(tgt, sents)
            r1 = score["rouge1"].fmeasure
            r2 = score["rouge2"].fmeasure
            rl = score["rougeL"].fmeasure
            rouge1 += r1
            rouge2 += r2
            rougeLsum += rl
            cnt += 1
            # print(rouge1/cnt, rouge2/cnt, rougeLsum/cnt)

    model.train()
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt

    logger.info(f"rouge-1: {rouge1}, rouge-2: {rouge2}, rouge-L: {rougeLsum}")
    return rouge1

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

    tokenizer = BertTokenizer.from_pretrained('../temp/bert-base-vocab.txt')

    def train_iter_fct():
        return Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device, shuffle=True, is_test=False, tokenizer=tokenizer)

    train_iter = train_iter_fct()
    total_steps = int(len(train_iter.cur_dataset) * args.epoch / args.accumulate_step)

    # 模型，优化器
    model = sort_model(args, device, tokenizer)
    model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)

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
            src_input = batch.src_input
            sent_input = batch.sent_input
            src_sent_labels = batch.src_sent_labels

            output = model(src_input, sent_input)
            doc, summary = output['doc_emb'], output['summary_emb']
            loss = doc_summ_loss(doc, summary, src_sent_labels)
            avg_loss += loss.item()
            loss.backward()

            step_cnt += 1
            if step_cnt == args.accumulate_step:

                step_cnt = 0
                sim_step += 1
                all_step_cnt += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if sim_step % args.report_freq == 0 and step_cnt == 0:
                logger.info("epoch: %d, batch: %d, avg loss: %.12f" % (epoch+1, sim_step, avg_loss / (args.accumulate_step * args.report_freq)))
                avg_loss = 0

            if all_step_cnt % 1 == 0 and all_step_cnt != 0 and step_cnt == 0:
                avg_r1 = test_abs(model, args, tokenizer)
                if avg_r1 > max_r1:
                    max_r1 = avg_r1
                    torch.save(model.state_dict(), os.path.join(args.model_save, str(all_step_cnt) + '.bin'))
                    logger.info("best - epoch: %d, all_step_cnt: %d" % (epoch, all_step_cnt))
                    logger.info("max avg_r1: %.6f" % (max_r1))

        train_iter = train_iter_fct()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', choices=['train', 'test'], type=str)
    parser.add_argument("-data_path", default='../data/cnndm/simtest', type=str)
    # parser.add_argument("-data_path", default='../bert_data/cnndm/doc_sum', type=str)
    parser.add_argument("-result_path", default='../results/', type=str)
    parser.add_argument("-temp_dir", default='../Roberta/', type=str)
    parser.add_argument("-log_file", default='../logs/train.txt', type=str)
    parser.add_argument("-model_save", default='../models/sort', type=str)
    parser.add_argument("-test_model_path", default='../models/sort/300.bin', type=str)

    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-scale", default=1, type=float)
    parser.add_argument("-margin", default=0.01, type=float)
    parser.add_argument("-gold_margin", default=0, type=float)
    parser.add_argument("-gold_weight", default=1, type=float)
    parser.add_argument("-lamda", default=0.05, type=float)

    parser.add_argument("-large", default=False, type=bool)
    parser.add_argument("-lr", default=3e-5, type=float)
    parser.add_argument("-warmup_proportion", default=0.1, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    parser.add_argument("-damping", default=0.8, type=float)
    parser.add_argument("-epoch", default=3, type=int)
    parser.add_argument("-accumulate_step", default=1, type=int)  # 12
    parser.add_argument("-grad_norm", default=0, type=int)
    parser.add_argument("-report_freq", default=2, type=int)

    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-seed', default=42, type=int)

    # decoder
    parser.add_argument('-doc_dec_layers', default=3, type=int)
    parser.add_argument('-doc_dec_hidden_size', default=768, type=int)
    parser.add_argument('-doc_dec_heads', default=8, type=int)
    parser.add_argument('-doc_dec_ff_size', default=2048, type=int)
    parser.add_argument('-doc_dec_dropout', default=0.3, type=int)


    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1


    if (args.mode == 'train'):
        train_abs(args, device_id)
    else:
        tokenizer = BertTokenizer.from_pretrained('../temp/bert-base-vocab.txt', cache_dir=None)
        checkpoint = torch.load(args.test_model_path)
        model = sort_model(args, device, tokenizer)
        model.load_state_dict(checkpoint, strict=True)
        model.to(device)
        test_abs(model, args, tokenizer)

