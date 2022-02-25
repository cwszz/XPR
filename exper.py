import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import math
from sklearn.model_selection import *
from transformers import AutoTokenizer, AutoConfig, BertTokenizer, BertConfig, BertModel, \
    XLMRobertaModel, XLMRobertaTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers.models import xlm_roberta

from DictMatching.Loss import NTXentLoss, MoCoLoss
from utilsWord.args import getArgs
from utilsWord.tools import seed_everything, AverageMeter
from utilsWord.sentence_process import load_words_mapping, WordWithContextDatasetWW, load_word2context_from_tsv_hiki


class BackBone_Model(nn.Module):

    def __init__(self, model='bert-base-uncased', layer_id=8, is_type=True, wo_linear_head=False):
        super(BackBone_Model, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.config = BertConfig.from_pretrained(model)
        self.model = BertModel.from_pretrained(model)
        self.is_type = is_type
        if self.is_type is True:
            self.model.embeddings.token_type_embeddings = nn.Embedding(
                2, self.config.hidden_size)
            self.model.embeddings.token_type_embeddings.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        self.wo_linear_head = wo_linear_head
        self.criterion = nn.CrossEntropyLoss()
        self.layer_id = layer_id

    # 就是把entity 提取出来成[1*H]的，同时把句子中的entity同样替换[1*H]
    def convert(self, opt, mask, index, length):
        '''obtain the opt and word position information (index, length),
        return tensor with shape [B, H]'''
        # opt: [B, S, H]; index: [B]; length: [B]; mask: [B, S]
        word_embd, new_opt, new_mask = [], [], []
        for item, item_mask, idx, l in zip(opt, mask, index, length):
            word_embd.append(item[idx:idx+l, :].mean(dim=0))    # [S, H] -> [H]
            if l > 1:
                item = torch.cat(
                    (
                        item[:idx, :],     # [S_, H]
                        word_embd[-1].unsqueeze(0),     # [1, H]
                        item[idx+l:, :],     # [S_, H]
                        torch.zeros(
                            l-1, self.config.hidden_size).half().to(item_mask.device)
                    ),
                    dim=0
                )
                item_mask = torch.cat(
                    (
                        item_mask[:idx],
                        item_mask[idx:idx+1],
                        item_mask[idx+l:],
                        torch.zeros(l-1).long().to(item_mask.device)
                    )
                )
            new_opt.append(item)   # [S', H]
            new_mask.append(item_mask)
        word_embd = torch.stack(word_embd)    # [B, H]
        new_opt = torch.stack(new_opt)
        new_mask = torch.stack(new_mask)
        # word_embd: [B, H]
        # new_opt: [B, S, H]
        # new_mask: [B, S]
        # lt
        # return word_embd, new_opt, new_mask
        return word_embd

    def extra_first_cls(self, opt):
        word_embd = []
        for item in opt:
            word_embd.append(item[0])
        word_embd = torch.stack(word_embd)
        return word_embd

    def extra_cls(self, opt, mask, index, length):
        word_embd = []
        for item, idx in zip(opt, index):
            word_embd.append(item[idx])
        word_embd = torch.stack(word_embd)
        return word_embd

    def _encode(self, input_ids, mask, type_ids):
        if self.is_type:
            all_layers_hidden = self.model(
                input_ids=input_ids, attention_mask=mask, token_type_ids=type_ids, output_hidden_states=True)[2]
        else:
            all_layers_hidden = self.model(
                input_ids=input_ids, attention_mask=mask, token_type_ids=None, output_hidden_states=True)[2]
        return all_layers_hidden[self.layer_id]

        # 先给一个id，mask 编码，然后 self.head去映射。w1_index 应该是该词的坐标，w1_length 应该是这个词的长度, 因为是 bpe好像就是每个词一个embedding。
        # 词的长度还需要想一想，感觉可能有问题
    def forward(self, w_indices=None, w_mask=None, w_index=None, w_length=None, w_type=None, sample_num=None):
        if sample_num != 0:
            '''w*_indices/w*_mask: [B, S]; w*_index/w*_length: [B]'''
            if self.wo_linear_head:
                opt_w = self._encode(
                    w_indices, w_mask, w_type)  # w/o linear_head
            else:
                opt_w = self.linear_head(self._encode(
                    w_indices, w_mask, w_type))    # [B, S1, H]
            # w1_embd = self.extra_first_cls(opt_w)
            # w1_embd = self.extra_cls(opt_w, w_mask, w_index, w_length)
            w1_embd = self.convert(
                opt_w, w_mask, w_index, w_length)   # [B, H] 平均提取
            w1_embd = torch.mean(
                w1_embd.reshape(-1, sample_num, w1_embd.size(-1)), dim=1)
        else:  # 此时进来的是word
            if self.wo_linear_head:
                w1_embd = torch.mean(self._encode(
                    w_indices, w_mask, w_type), dim=1)
                # w1_embd = self._encode(w_indices,w_mask,w_type)[:,0,:] # w/o linear_head
            else:
                # w1_embd = self.linear_head(self._encode(w_indices,w_mask,w_type))[:,0,:]
                w1_embd = torch.mean(self.linear_head(
                    self._encode(w_indices, w_mask, w_type)), dim=1)

        return w1_embd


args = getArgs()
num = 0
seed_everything(args.seed)  # 固定随机种子
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
if args.distributed:
    device = torch.device('cuda', args.local_rank)
else:

    device = torch.device('cuda:{}'.format(str(num)))
    torch.cuda.set_device(num)
lossFunc = MoCoLoss().to(device)


def test_model(model, val_loader):  # 验证
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        first_src_examples, first_trg_examples = None, None
        second_src_examples, second_trg_examples = None, None
        for step, batch in enumerate(tk):
            batch_src = [tensors.to(device)
                         for i, tensors in enumerate(batch) if i % 2 == 0]
            batch_trg = [tensors.to(device)
                         for i, tensors in enumerate(batch) if i % 2 == 1]
            if args.distributed:
                first_src = model.module.encoder_q(
                    *batch_src, sample_num=args.dev_sample_num)
                first_trg = model.module.encoder_q(
                    *batch_trg, sample_num=args.dev_sample_num)
                second_src = model.module.encoder_q(
                    *batch_src, sample_num=args.dev_sample_num)
                second_trg = model.module.encoder_q(
                    *batch_trg, sample_num=args.dev_sample_num)
            else:
                first_src = model.encoder_q(
                    *batch_src, sample_num=args.dev_sample_num)
                first_trg = model.encoder_q(
                    *batch_trg, sample_num=args.dev_sample_num)
                second_src = model.encoder_q(
                    *batch_src, sample_num=args.dev_sample_num)
                second_trg = model.encoder_q(
                    *batch_trg, sample_num=args.dev_sample_num)
            first_src_examples = first_src if first_src_examples is None else torch.cat(
                [first_src_examples, first_src], dim=0)
            first_trg_examples = first_trg if first_trg_examples is None else torch.cat(
                [first_trg_examples, first_trg], dim=0)
            second_src_examples = second_src if second_src_examples is None else torch.cat(
                [second_src_examples, second_src], dim=0)
            second_trg_examples = second_trg if second_trg_examples is None else torch.cat(
                [second_trg_examples, second_trg], dim=0)
        first_src_examples = torch.nn.functional.normalize(
            first_src_examples, dim=1)
        first_trg_examples = torch.nn.functional.normalize(
            first_trg_examples, dim=1)
        second_src_examples = torch.nn.functional.normalize(
            second_src_examples, dim=1)
        second_trg_examples = torch.nn.functional.normalize(
            second_trg_examples, dim=1)
        first_st_sim_matrix = F.softmax(torch.mm(
            first_src_examples, first_trg_examples.T)/math.sqrt(first_src_examples.size(-1))/0.1, dim=1)
        second_st_sim_matrix = F.softmax(torch.mm(
            second_trg_examples, second_src_examples.T)/math.sqrt(second_trg_examples.size(-1))/0.1, dim=1)
        label = torch.LongTensor(list(range(first_st_sim_matrix.size(0)))).to(
            first_src_examples.device)
        st_acc = torch.argmax(first_st_sim_matrix, dim=1)    # [B]
        ts_acc = torch.argmax(second_st_sim_matrix, dim=1)
        acc = (st_acc == label).long().sum().item() / st_acc.size(0)
        acc += (ts_acc == label).long().sum().item() / ts_acc.size(0)
    return acc / 2


"""
模型训练数据准备
"""
if args.distributed:
    dist.init_process_group(backend='nccl')

"""
模型训练预测
"""
if __name__ == '__main__':
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    config = AutoConfig.from_pretrained('xlm-roberta-base')
    model = XLMRobertaModel.from_pretrained('xlm-roberta-base').to(device)
    input_one = tokenizer("故宫", return_tensors="pt").to(device)
    input_two = tokenizer("Macbook", return_tensors="pt").to(device)
    input_three = tokenizer("笔记本电脑", return_tensors="pt").to(device)
    input_four = tokenizer("Computer", return_tensors="pt").to(device)
    input_five = tokenizer("my phone", return_tensors="pt").to(device)
    output_one = model(**input_one, output_hidden_states=True)[2]
    output_two = model(**input_two, output_hidden_states=True)[2]
    output_three = model(**input_three, output_hidden_states=True)[2]
    output_four = model(**input_four, output_hidden_states=True)[2]
    output_five = model(**input_five, output_hidden_states=True)[2]
    for each_layer in range(12):
        representation_one = torch.mean(
            output_one[each_layer].squeeze(0), dim=0).unsqueeze(0)
        representation_two = torch.mean(
            output_two[each_layer].squeeze(0), dim=0).unsqueeze(0)
        representation_three = torch.mean(
            output_three[each_layer].squeeze(0), dim=0).unsqueeze(0)
        representation_four = torch.mean(
            output_four[each_layer].squeeze(0), dim=0).unsqueeze(0)
        representation_five = torch.mean(
            output_five[each_layer].squeeze(0), dim=0).unsqueeze(0)
        one = torch.nn.functional.normalize(representation_one, dim=1)
        two = torch.nn.functional.normalize(representation_two, dim=1)
        three = torch.nn.functional.normalize(representation_three, dim=1)
        four = torch.nn.functional.normalize(representation_four, dim=1)
        five = torch.nn.functional.normalize(representation_five, dim=1)
        sim1 = torch.mm(one, two.T)
        sim2 = torch.mm(three, two.T)
        sim3 = torch.mm(four, two.T)
        sim4 = torch.mm(five, two.T)
        sim = torch.mm(four, three.T)
        print(sim1.item())
        print(sim2.item())
        print(sim3.item())
        print(sim4.item())
        print("-------")
    print(output_two)
