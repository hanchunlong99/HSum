import copy

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
from models.decoder import TransformerDecoder
from models.optimizers import Optimizer
from models.encoder import Classifier, ExtTransformerEncoder, TransformerEncoderLayer
from transformers import XLNetModel
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]

    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device):  # .. 768 cuda
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.config = BertConfig.from_pretrained('../temp/bert/bert-base-config.json')
            self.model = BertModel.from_pretrained('../temp/bert/bert-base-pytorch_model.bin', config=self.config)

        self.finetune = finetune

    def forward(self, x, segs, mask):

        if(self.finetune):
            top_vec = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec = self.model(x, segs, attention_mask=mask)
        return top_vec

class XLNet(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(XLNet, self).__init__()
        if (large):
            self.model = XLNetModel.from_pretrained('xlnet-large-cased', cache_dir=temp_dir)
            # self.model.config.output_attentions = True
        else:
            print("base")
            self.model = XLNetModel.from_pretrained('xlnet-base-cased', cache_dir=temp_dir)
            self.model.config.output_attentions = True
        self.finetune = finetune

    def forward(self, x, segs, mask):

        if (self.finetune):
            top_vec = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        return top_vec

# XLNet
class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.value = nn.Linear(768, 768)
        self.wo = nn.Linear(768, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.xlnet = XLNet(args.large, args.temp_dir, args.finetune_bert)
        self.ext_layer = ExtTransformerEncoder(self.xlnet.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)
    #
    # def forward(self, src, segs, clss, mask_src, mask_cls):
    #     top_vec = self.xlnet(src, mask_src, segs)[0]
    #     sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
    #     sents_vec = sents_vec * mask_cls[:, :, None].float()
    #     sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
    #     return sent_scores, mask_cls

    def forward(self, src, segs, clss, mask_src, mask_cls):
        # src, segs, clss, mask, mask_cls, sent_sign

        top_vec = self.xlnet(src, segs, mask_src+0)

        sent_embedding = top_vec[0]  # batch * 512 * 768
        att = top_vec['attentions'][11]  #
        # 先合并注意力特征矩阵，再选句子做评分
        # value矩阵的生成方式，合并注意力
        mixed_value_layer = self.value(sent_embedding)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        context_layer = torch.matmul(att, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (768,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 拿句子做评分
        context_layer = context_layer[torch.arange(sent_embedding.size(0)).unsqueeze(1), clss]
        sent_scores1 = self.sigmoid(self.wo(self.LayerNorm(context_layer)))
        sent_scores1 = sent_scores1.squeeze(-1) * mask_cls.float()

        return sent_scores1, mask_cls

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (12, 64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

# class ExtSummarizer(nn.Module):
#     def __init__(self, args, device, checkpoint):
#         super(ExtSummarizer, self).__init__()
#         self.args = args
#         self.device = device
#         self.value = nn.Linear(768, 768)
#         self.wo = nn.Linear(768, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()
#         # self.dropout = nn.Dropout(0.2)
#         self.LayerNorm = BertLayerNorm(768, eps=1e-12)
#         # self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
#         self.xlnet = XLNet(args.large, args.temp_dir, args.finetune_bert)
#         self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
#                                                args.ext_dropout, args.ext_layers)
#         if (args.encoder == 'baseline'):
#             bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
#                                      num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
#             self.bert.model = BertModel(bert_config)
#             self.ext_layer = Classifier(self.bert.model.config.hidden_size)
#
#         if(args.max_pos>512):
#             my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
#             my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
#             my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
#             self.bert.model.embeddings.position_embeddings = my_pos_embeddings
#
#
#         if checkpoint is not None:
#             self.load_state_dict(checkpoint['model'], strict=True)
#         else:
#             if args.param_init != 0.0:
#                 for p in self.ext_layer.parameters():
#                     p.data.uniform_(-args.param_init, args.param_init)
#             if args.param_init_glorot:
#                 for p in self.ext_layer.parameters():
#                     if p.dim() > 1:
#                         xavier_uniform_(p)
#
#         self.to(device)
#
#     def forward(self, src, segs, clss, mask_src, mask_cls):
#         top_vec = self.bert(src, segs, mask_src)[0]
#         sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
#         sents_vec = sents_vec * mask_cls[:, :, None].float()
#         sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
#         return sent_scores, mask_cls
#
#     def forward(self, src, segs, clss, mask_src, mask_cls):
#         # src, segs, clss, mask, mask_cls, sent_sign
#         top_vec = self.bert(src, segs, mask_src)
#         sent_embedding = top_vec[0]
#         att = top_vec[2][11]
#         # 先合并注意力特征矩阵，再选句子做评分
#
#         # value矩阵的生成方式，合并注意力
#         mixed_value_layer = self.value(sent_embedding)
#         value_layer = self.transpose_for_scores(mixed_value_layer)
#         context_layer = torch.matmul(att, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (768,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#
#         # 拿句子做评分
#         context_layer = context_layer[torch.arange(sent_embedding.size(0)).unsqueeze(1), clss]
#         sent_scores1 = self.sigmoid(self.wo(self.LayerNorm(context_layer)))
#         sent_scores1 = sent_scores1.squeeze(-1) * mask_cls.float()
#
#         return sent_scores1, mask_cls
#
#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (12, 64)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)
#
#
#
# XLNet

# class AbsSummarizer(nn.Module):
#     def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
#         super(AbsSummarizer, self).__init__()
#         self.args = args
#         self.device = device
#         self.xlnet = XLNet(args.large, args.temp_dir, args.finetune_bert)
#
#         if bert_from_extractive is not None:
#             self.bert_2.model.load_state_dict(
#                 dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)
#
#         if (args.encoder == 'baseline'):
#             bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
#                                      num_hidden_layers=args.enc_layers, num_attention_heads=8,
#                                      intermediate_size=args.enc_ff_size,
#                                      hidden_dropout_prob=args.enc_dropout,
#                                      attention_probs_dropout_prob=args.enc_dropout)
#             self.bert.model = BertModel(bert_config)
#
#         self.vocab_size = self.xlnet.model.config.vocab_size
#
#         tgt_embeddings = nn.Embedding(self.vocab_size, self.xlnet.model.config.hidden_size, padding_idx=0)
#         if (self.args.share_emb):
#             tgt_embeddings.weight = copy.deepcopy(self.xlnet.model.embeddings.word_embeddings.weight)
#
#         # 知识复用
#         # self.decoder = TransformerDecoder(
#         #     self.args.dec_layers,
#         #     self.args.dec_hidden_size, heads=self.args.dec_heads,
#         #     d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings, ext_model=args.ext_model)
#
#         self.decoder = TransformerDecoder(
#             self.args.dec_layers,
#             self.args.dec_hidden_size, heads=self.args.dec_heads,
#             d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
#
#         self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
#         self.generator[0].weight = self.decoder.embeddings.weight
#
#
#         if checkpoint is not None:
#             self.load_state_dict(checkpoint['model'], strict=True)
#         else:
#             for module in self.decoder.modules():
#                 if isinstance(module, (nn.Linear, nn.Embedding)):
#                     module.weight.data.normal_(mean=0.0, std=0.02)
#                 elif isinstance(module, nn.LayerNorm):
#                     module.bias.data.zero_()
#                     module.weight.data.fill_(1.0)
#                 if isinstance(module, nn.Linear) and module.bias is not None:
#                     module.bias.data.zero_()
#             for p in self.generator.parameters():
#                 if p.dim() > 1:
#                     xavier_uniform_(p)
#                 else:
#                     p.data.zero_()
#             if (args.use_bert_emb):
#                 tgt_embeddings = nn.Embedding(self.vocab_size, self.xlnet.model.config.hidden_size, padding_idx=0)
#                 # print(self.xlnet.model)
#                 tgt_embeddings.weight = copy.deepcopy(self.xlnet.model.word_embedding.weight)
#                 self.decoder.embeddings = tgt_embeddings
#                 self.generator[0].weight = self.decoder.embeddings.weight
#
#         self.to(device)
#
#     def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
#
#         top_vec = self.xlnet(src, segs, mask_src+0)[0]
#         dec_state = self.decoder.init_decoder_state(src, top_vec)
#         decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
#
#         return decoder_outputs, None


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)[0]
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
