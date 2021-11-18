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
import random
import os
import time
from sklearn.model_selection import *
from transformers import AdamW, get_cosine_schedule_with_warmup, BertConfig, \
    RobertaConfig, get_linear_schedule_with_warmup, \
    RobertaTokenizerFast, BertTokenizerFast, AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from DictMatching.Loss import NTXentLoss, MoCoLoss
from DictMatching.SimCLR import SimCLR
from DictMatching.data.entity.readTsv import getEval
from DictMatching.data.train.readEnWords import cleanDictionary
from DictMatching.moco import MoCo
from utilsWord.args import getArgs
from utilsWord.process import CustomDataset
from utilsWord.tools import seed_everything, AverageMeter
from utilsWord.sentence_process import load_words_mapping,WordWithContextDatasetWW,load_word2context_from_tsv_hiki

args = getArgs()
num = 4
seed_everything(args.seed)  # 固定随机种子
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
if args.distributed:
    device = torch.device('cuda', args.local_rank)
else:
    
    device = torch.device('cuda:{}'.format(str(num)))
    torch.cuda.set_device(num)
lossFunc = MoCoLoss().to(device)

def train_model(model, train_loader):  # 训练一个epoch
    scaler = GradScaler()
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    clips = AverageMeter()

    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    for step, batch in enumerate(tk):
        batch_src = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 0]
        batch_trg = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 1]
        output0, output1 = model(batch_src,batch_trg)
        loss1, acc1 = lossFunc(output0, output1)
        output0, output1 = model(batch_trg,batch_src)
        loss2, acc2 = lossFunc(output0, output1)
        loss = loss1 + loss2
        acc = (acc1 + acc2)/2
        loss = loss / 2

        input_ids = batch_src[0]
        scaler.scale(loss).backward()
        # loss.backward()

        clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clips.update(clip.item(), input_ids.size(0))

        losses.update(loss.item(), input_ids.size(0))
        accs.update(acc, input_ids.size(0))

        tk.set_postfix(loss=losses.avg, acc=accs.avg, clips=clips.avg)

        if ((step + 1) % args.gradient_accumulation_steps == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

    return losses.avg, accs.avg


def test_model(model, val_loader):  # 验证
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()
    

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        first_src_examples,first_trg_examples = None,None
        second_src_examples,second_trg_examples = None,None
        for step, batch in enumerate(tk):
            batch_src = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 0]
            batch_trg = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 1]
            if args.distributed:
                first_src = model.module.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                first_trg = model.module.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
                second_src = model.module.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                second_trg = model.module.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
            else:
                first_src = model.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                first_trg = model.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
                second_src = model.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                second_trg = model.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
            first_src_examples = first_src if first_src_examples is None else torch.cat([first_src_examples,first_src],dim=0)
            first_trg_examples = first_trg if first_trg_examples is None else torch.cat([first_trg_examples,first_trg],dim=0)
            second_src_examples = second_src if second_src_examples is None else torch.cat([second_src_examples,second_src],dim=0)
            second_trg_examples = second_trg if second_trg_examples is None else torch.cat([second_trg_examples,second_trg],dim=0)
        first_src_examples = torch.nn.functional.normalize(first_src_examples,dim=1)
        first_trg_examples = torch.nn.functional.normalize(first_trg_examples,dim=1)
        second_src_examples = torch.nn.functional.normalize(second_src_examples,dim=1)
        second_trg_examples = torch.nn.functional.normalize(second_trg_examples,dim=1)
        first_st_sim_matrix = F.softmax(torch.mm(first_src_examples,first_trg_examples.T)/math.sqrt(first_src_examples.size(-1))/0.1,dim=1)
        second_st_sim_matrix = F.softmax(torch.mm(second_trg_examples,second_src_examples.T)/math.sqrt(second_trg_examples.size(-1))/0.1,dim=1)
        label = torch.LongTensor(list(range(first_st_sim_matrix.size(0)))).to(first_src_examples.device)
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
    test_dev = args.test_dev
    if test_dev:
        args.test_entity_path = "./our_dataset/dev/dev-en-" + args.test_lg + "-" + args.sn + "-entity.txt"
    else:
        args.test_entity_path = "./our_dataset/test/test-en-" + args.test_lg + "-" + args.sn + "-entity.txt"

    args.src_context_path = "./our_dataset/sentences/en-" + args.test_lg + "-entity-sentences." + args.sn + ".tsv"
    args.trg_context_path =  "./our_dataset/sentences/" + args.test_lg + "-entity-sentences." +args.sn + ".tsv"

    quene_length = int(args.quene_length)
    para_T = args.T_para
    args.output_model_path = './' + args.output_log_dir+ '/' + str(args.train_sample_num) + '-' + args.lg+ '-'+ str(args.all_sentence_num)  + '-' +args.wo_span_eos + '-' + str(quene_length) + '-' + str(para_T) + '-' + str(args.seed) \
        + '-' + str(args.num_train_epochs)  + '-' + str(args.momentum) + '-' + str(args.simclr) + '-dev_qq-layer_12' +  '/best.pt'
    print(args.output_model_path)
    test = load_words_mapping(args.test_entity_path)
    en_word2context = load_word2context_from_tsv_hiki(args.src_context_path,args.dev_all_sentence_num)
    lg_word2context = load_word2context_from_tsv_hiki(args.trg_context_path,args.dev_all_sentence_num)
    unsup = args.unsupervised
    
    test_dataset = WordWithContextDatasetWW(test, en_word2context, lg_word2context,sampleNum=args.dev_sample_num,
        max_len=args.sentence_max_len,cut_type=args.cut_type)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate, shuffle=False,num_workers=16)

    best_acc = 0

    """
    分层训练 自适应模型
    模型训练准备 参数设定
    """
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = MoCo(config=config,K=quene_length,T=para_T,args=args).to(device)
    
        
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)
    if not unsup:
        model.load_state_dict(torch.load(args.output_model_path,map_location={'cuda:1':'cuda:0'}))  # 
    val_acc = test_model(model, test_loader)
    print("src-lg: " + args.lg  +" trg-lg: " + args.test_lg + " acc:", val_acc)
