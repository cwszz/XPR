import torch
import torch.nn as nn
from transformers import XLMRobertaModel,XLMRobertaConfig,XLMRobertaTokenizer
from transformers.tokenization_utils import AddedToken
from torch.cuda.amp import autocast,GradScaler
from DictMatching.SimCLR import _get_simclr_projection_head



class BackBone_Model(nn.Module):
    
    def __init__(self, model='xlm-roberta-base', layer_id=8,is_type=True):
        super(BackBone_Model, self).__init__()
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model)
        self.config = XLMRobertaConfig.from_pretrained(model)
        self.model = XLMRobertaModel.from_pretrained(model)
        self.is_type = is_type
        if self.is_type is True:
            self.model.embeddings.token_type_embeddings = nn.Embedding(2,self.config.hidden_size)
            self.model.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.linear_head = _get_simclr_projection_head(self.config.hidden_size, self.config.hidden_size)
        self.criterion = nn.CrossEntropyLoss()
        self.layer_id = layer_id
        
    def convert(self, opt, mask, index, length):  # 就是把entity 提取出来成[1*H]的，同时把句子中的entity同样替换[1*H]
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
                        torch.zeros(l-1, self.config.hidden_size).half().to(item_mask.device)
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
        return word_embd
    
    def extra_first_cls(self,opt):
        word_embd = []
        for item in opt:
            word_embd.append(item[0])
        word_embd = torch.stack(word_embd)
        return word_embd

    def extra_cls(self,opt, mask, index, length):
        word_embd = []
        for item, idx in zip(opt, index):
            word_embd.append(item[idx])
        word_embd = torch.stack(word_embd)  
        return word_embd

    def _encode(self, input_ids, mask,type_ids):
        if self.is_type:
            all_layers_hidden = self.model(input_ids=input_ids, attention_mask=mask,token_type_ids=type_ids,output_hidden_states=True)[2]
        else:
            all_layers_hidden = self.model(input_ids=input_ids, attention_mask=mask,token_type_ids=None,output_hidden_states=True)[2]
        return all_layers_hidden[self.layer_id]
        
        # 先给一个id，mask 编码，然后 self.head去映射。w1_index 应该是该词的坐标，w1_length 应该是这个词的长度, 因为是 bpe好像就是每个词一个embedding。
        # 词的长度还需要想一想，感觉可能有问题
    def forward(self, w_indices=None, w_mask=None, w_index=None, w_length=None,w_type=None,sample_num=None):
        if sample_num != 0:
            '''w*_indices/w*_mask: [B, S]; w*_index/w*_length: [B]'''
            opt_w = self.linear_head(self._encode(w_indices, w_mask,w_type))    # [B, S1, H]
            # w1_embd = self.extra_first_cls(opt_w)
            # w1_embd = self.extra_cls(opt_w, w_mask, w_index, w_length)
            w1_embd = self.convert(opt_w, w_mask, w_index, w_length)   # [B, H] 平均提取
            w1_embd = torch.mean(w1_embd.reshape(-1,sample_num,w1_embd.size(-1)),dim=1)
        else: # 此时进来的是word
            w1_embd = self.linear_head(self._encode(w_indices,w_mask,w_type))[:,0,:]
            # w1_embd = torch.mean(self.linear_head(self._encode(w_indices,w_mask,w_type)),dim=1)
        
        return w1_embd


class MoCo_simclr(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self,args, T=0.04,config=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_simclr, self).__init__()
        self.train_sample_num = args.train_sample_num
        self.is_distributed = args.distributed
        self.T = T

        self.encoder_q = BackBone_Model(layer_id=args.layer_id,is_type=True if args.is_type == 1 else False)


    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        with autocast():
            q = self.encoder_q(*im_q,sample_num=self.train_sample_num)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)

            k = self.encoder_q(*im_k,sample_num=self.train_sample_num)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            logits = torch.einsum('nc,mc->nm', [q, k])

            logits /= self.T

        # labels: positive key indicators
            labels = torch.tensor([i for i in range(logits.shape[0])], dtype=torch.long).to(im_q[0].device)

            return logits, labels


