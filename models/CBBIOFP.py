import torch
import torch.nn as nn
from models.seqencoder import *

class CBBIOMFP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args           = args
        self.precet         = args.ds.model.precet
        self.dropout        = args.ds.model.dropout
        
        self.src_emb        = nn.Embedding(
        num_embeddings      = args.ds.encoder.vocab_size,
        embedding_dim       = args.ds.encoder.emb_dim).cuda()
        
        self.pos_emb        = PositionalEncoding(
        d_model             = args.ds.encoder.emb_dim).cuda()
        
        self.attention      = Encoder(
        n_heads             = 2,
        n_layers            = 1).cuda()
        
        self.trans_encoder  = Encoder(
        n_heads             = 2,
        n_layers            = 2).cuda()
        
        self.encoder        = Encoder().cuda()

        self.pre_head       = nn.Sequential(
                                nn.Linear(args.ds.encoder.emb_dim * args.ds.encoder.max_len, 1024),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(512, 128)).cuda()
        
    def forward(self, peptide):
        enc_outputs = self.src_emb(peptide)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        _, mask_self_attn, _ = self.attention(peptide, enc_outputs)
        lenth = len(mask_self_attn)
        layer_sum = mask_self_attn[0]
        flag = 1
        while lenth != 1:
            layer_sum = layer_sum + mask_self_attn[flag]
            flag += 1
            lenth -= 1

        head_sum = layer_sum.sum(dim=1)
        mask = head_sum.sum(dim=1)

        mask_peptide = creat_mask_matrix(mask, peptide, self.precet)

        enc_outputs_mask = self.src_emb(mask_peptide)
        enc_outputs_mask = self.pos_emb(enc_outputs_mask.transpose(0, 1)).transpose(0, 1)

        enc_outputs, enc_self_attns, attn_score = self.encoder(peptide, enc_outputs)
        enc_outputs_mask, enc_self_attns_mask, _ = self.trans_encoder(mask_peptide, enc_outputs_mask)

        h1 = torch.reshape(enc_outputs, (enc_outputs.shape[0], -1))
        h2 = torch.reshape(enc_outputs_mask, (enc_outputs_mask.shape[0], -1))

        z1 = self.pre_head(h1)
        z2 = self.pre_head(h2)

        return h1, z1, h2, z2, attn_score


def creat_mask_matrix(mask, peptide, precet):
    """mask (batch_size, src_len) is matrix sumed
    peptide (batch_size, src_len) is input
    this function is to creat a mask_matrix"""
    
    raw = peptide.shape[0]
    line = peptide.shape[1]
    one = torch.ones(raw, line).int().cuda()
    for i in range(raw):
        for j in range(line):
            if mask[i][j] == 0 or j == line:
                values, indices = torch.topk(mask[i][:j], int(math.ceil(precet * j)), largest=False, sorted=False)
                one[i][indices] = 0
                break
            else:
                continue
    re = peptide.mul(one)
    return re