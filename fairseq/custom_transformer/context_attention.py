import math
import copy

import torch, torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class SequenceGating(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.fc_q = nn.Linear(embed_dim, 1)
        self.fc_k = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k):
        alpha = torch.sigmoid(self.dropout(self.fc_q(q) + self.fc_k(k))) # batch_size, seq_len, 1
        x = (1-alpha) * q + alpha * k # batch_size, seq_len, dim
        return x

class ContextAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.encoder.embed_dim
        self.heads = cfg.encoder.attention_heads
        self.head_dim = self.embed_dim // self.heads
        self.eps = 1e-6
        self.linears = clones(nn.Linear(self.embed_dim, self.embed_dim), 4)
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.context_linears = clones(nn.Linear(self.embed_dim, self.embed_dim), 2)
        self.query_gating = SequenceGating(self.head_dim)
        self.key_gating = SequenceGating(self.head_dim)
         
    def compute_global_vector(self, x, mask=None):
        if mask is not None:
            input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        else:
            input_mask_expanded = torch.ones_like(x).float()
        sum_embeddings = torch.sum(x * input_mask_expanded, -2)
        sum_mask = input_mask_expanded.sum(-2)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        global_embeddings = sum_embeddings / sum_mask
        return global_embeddings

    def forward(self, x, mask):
        x = x.transpose(0, 1)
        batch_size = x.size(0)

        # query = key = value = batch_size, seq_len, dim
        query, key, value = \
            [l(vector).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
             for l, vector in zip(self.linears, (x, x, x))] 

        # global_embeds = (batch_size, 1, dim) => expand to (batch_size, seq_len, dim)
        global_embeds = self.compute_global_vector(x, mask).unsqueeze(1)
        global_query, global_key = [l(vector).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2).expand(query.size())
             for l, vector in zip(self.context_linears, (global_embeds, global_embeds))]

        query_hat = self.query_gating(query, global_query)
        key_hat = self.key_gating(key, global_key)
        
        x, attn_score = attention(query_hat, key_hat, value, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
             .view(batch_size, -1, self.heads * self.head_dim)
        x = x.transpose(0, 1)
        return self.linears[-1](x), attn_score