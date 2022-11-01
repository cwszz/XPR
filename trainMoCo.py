import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import math
import os
from sklearn.model_selection import *
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig, XLMRobertaTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from DictMatching.Loss import MoCoLoss
# from DictMatching.simclr import MoCo_simclr
from DictMatching.moco import MoCo
from utilsWord.args import getArgs
from utilsWord.tools import seed_everything, AverageMeter
from utilsWord.sentence_process import load_words_mapping,WordWithContextDatasetWW, load_word2context_from_tsv

args = getArgs()
seed_everything(args.seed)  # 固定随机种子

if args.distributed:
    device = torch.device('cuda', args.local_rank)
else:
    num = 7
    device = torch.device('cuda:{}'.format(str(num)))
    torch.cuda.set_device(num)
lossFunc = MoCoLoss().to(device)

def train_model(model, train_loader):  # Train an epoch
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
        with autocast():
            output0, output1 = model(batch_src,batch_trg)
            loss1, acc1 = lossFunc(output0, output1)
            output0, output1 = model(batch_trg,batch_src)
            loss2, acc2 = lossFunc(output0, output1)
        loss = loss1 + loss2
        with open('./ano_record.txt','a+')  as f:
            f.write("STEP : " + str(step) + '\n')
            f.write(str(loss1) + " | "+  str(loss2) + '\n')
        acc = (acc1 + acc2)/2
        loss = loss / 2
        input_ids = batch_src[0]
        # loss.backward()
        scaler.scale(loss).backward()

        clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clips.update(clip.item(), input_ids.size(0))

        losses.update(loss.item(), input_ids.size(0))
        accs.update(acc, input_ids.size(0))

        tk.set_postfix(loss=losses.avg, acc=accs.avg)

        if ((step + 1) % args.gradient_accumulation_steps == 0) or ((step + 1) == len(train_loader)): 
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return losses.avg, accs.avg


def test_model_single_encoder(model, val_loader):
    model.eval()

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


def test_model_dual_encoder(model, val_loader):  # Dual Encoder
    model.eval()
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        first_src_examples,first_trg_examples = None,None
        second_src_examples,second_trg_examples = None,None
        for step, batch in enumerate(tk):
            batch_src = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 0]
            batch_trg = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 1]
            if args.distributed:
                first_src = model.module.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                first_trg = model.module.encoder_k(*batch_trg,sample_num=args.dev_sample_num)
                second_src = model.module.encoder_k(*batch_src,sample_num=args.dev_sample_num)
                second_trg = model.module.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
            else:
                first_src = model.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                first_trg = model.encoder_k(*batch_trg,sample_num=args.dev_sample_num)
                second_src = model.encoder_k(*batch_src,sample_num=args.dev_sample_num)
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


if args.distributed:
    dist.init_process_group(backend='nccl')

if __name__ == '__main__':
    # PARA
    args.train_phrase_path = "./data/train/train-en-" + args.lg + "-" + args.sn + "-phrase.txt"
    args.dev_phrase_path = "./data/dev/dev-en-" + args.lg + "-" + args.sn + "-phrase.txt"
    args.test_phrase_path = "./data/test/test-en-" + args.lg + "-" + args.sn + "-phrase.txt"
    args.src_context_path = "./data/sentences/en-" + args.lg + "-phrase-sentences." + args.sn + ".tsv"
    args.trg_context_path =  "./data/sentences/" + args.lg + "-phrase-sentences." +args.sn + ".tsv"
    queue_length = int(args.queue_length)
    para_T = args.T_para
    with_span_eos = True if args.wo_span_eos == 'true' else False
    dev_filename = '-dev_qq' if args.dev_only_q_encoder == 1 or args.simclr == 1  else '-dev_qk'
    wolinear = '-wolinear' if args.wolinear == 1 else ''
    args.output_loss_dir = './' + args.output_log_dir + '/' + str(args.train_sample_num) + '-' + args.lg+ '-'+str(args.all_sentence_num)+ '-' +args.wo_span_eos + '-' + str(queue_length) + '-' + str(para_T)  + '-' + str(args.seed) \
            + '-' + str(args.num_train_epochs) + '-' + str(args.momentum) + '-' + str(args.simclr) + dev_filename + '-layer_' + str(args.layer_id) + wolinear
    args.output_model_path = './' + args.output_log_dir+ '/' + str(args.train_sample_num) + '-' + args.lg+ '-'+str(args.all_sentence_num) + '-' +args.wo_span_eos + '-' + str(queue_length) + '-' + str(para_T) + '-' + str(args.seed) \
        + '-' + str(args.num_train_epochs)  + '-' + str(args.momentum) + '-' + str(args.simclr) + dev_filename + '-layer_' + str(args.layer_id) + wolinear +  '/best.pt'
    best_acc = 0
    # Data
    train_phrase_pairs = load_words_mapping(args.train_phrase_path)
    dev_phrase_pairs = load_words_mapping(args.dev_phrase_path)
    test_phrase_pairs = load_words_mapping(args.test_phrase_path)
    en_word2context = load_word2context_from_tsv(args.src_context_path,args.all_sentence_num)
    lg_word2context = load_word2context_from_tsv(args.trg_context_path,args.all_sentence_num)
    train_dataset = WordWithContextDatasetWW(train_phrase_pairs, en_word2context, lg_word2context,prepend_bos=with_span_eos,append_eos=with_span_eos,sampleNum=args.train_sample_num,
        max_len=args.sentence_max_len)
    dev_dataset = WordWithContextDatasetWW(dev_phrase_pairs, en_word2context, lg_word2context,prepend_bos=with_span_eos,append_eos=with_span_eos,sampleNum=args.dev_sample_num,
        max_len=args.sentence_max_len)
    test_dataset = WordWithContextDatasetWW(test_phrase_pairs, en_word2context, lg_word2context,prepend_bos=with_span_eos,append_eos=with_span_eos,sampleNum=args.dev_sample_num,
        max_len=args.sentence_max_len)
    
    # Data Loader
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=args.local_rank)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                              collate_fn=train_dataset.collate,drop_last=True,num_workers=16)
    val_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=dev_dataset.collate, shuffle=False,num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate, shuffle=False,num_workers=16)

    # Model Init
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    # if args.simclr == 1:
    #     model = MoCo_simclr(config=config,args=args,T=para_T).to(device)
    # else:
    model = MoCo(config=config,args=args,K=queue_length,T=para_T,m=args.momentum).to(device)

    bert_param_optimizer = model.named_parameters()
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, len(
        train_loader) // args.gradient_accumulation_steps,
                                                args.num_train_epochs * len(
                                                    train_loader) // args.gradient_accumulation_steps)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)
    else:
        model.to(device)

    if args.local_rank == 1 or args.local_rank == -1:
        print(device)
        print(args)
        print(model)
    # Test before train
    # if args.simclr == 1 or args.dev_only_q_encoder == 1:
    #     test_acc = test_model_single_encoder(model,test_loader)
    # else:
    #     test_acc = test_model_dual_encoder(model,test_loader)

    for epoch in range(args.num_train_epochs):

        print('epoch:', epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_model(model, train_loader)
        # if args.simclr == 1 or args.dev_only_q_encoder == 1:
        #     val_acc = test_model_single_encoder(model,val_loader)
        # else:
        #     val_acc = test_model_dual_encoder(model,val_loader)
        val_acc = test_model_single_encoder(model,val_loader)

        if args.local_rank == 1 or args.local_rank == -1:
            if not os.path.exists(args.output_loss_dir):
                os.mkdir(args.output_loss_dir)
            with open(args.output_loss_dir + '/loss_acc.txt','a+') as f:
                f.write("acc:{},best_acc:{}\n".format(str(val_acc),str(best_acc)))
            print("acc:", val_acc, "best_acc", best_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                if args.distributed:
                    torch.save(model.state_dict(),args.output_model_path)  # save as distributed
                else:
                    torch.save(model.state_dict(),args.output_model_path)
    if args.local_rank == 1 or args.local_rank == -1:
        model.load_state_dict(torch.load(args.output_model_path))
        val_acc = test_model_single_encoder(model,test_loader)
        # if args.simclr == 1 or args.dev_only_q_encoder == 1:
        #     val_acc = test_model_single_encoder(model,test_loader)
        # else:
        #     val_acc = test_model_dual_encoder(model,test_loader)
        with open(args.output_loss_dir + '/loss_acc.txt','a+') as f:
            f.write("TEST:  acc:{}\n".format(str(val_acc)))
            # Test before Train
            # f.write("Unsupervised TEST:  acc:{}\n".format(str(test_acc)))