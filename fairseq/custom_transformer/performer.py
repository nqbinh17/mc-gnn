import torch
import torch.nn as nn

from performer_pytorch import FastAttention

class Performer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.encoder.embed_dim
        self.heads = cfg.encoder.attention_heads
        assert self.embed_dim % self.heads == 0
        self.head_dim = self.embed_dim // self.heads
        self.eps = 1e-6

        self.fc_query = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_key = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_value = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_fn = FastAttention(
            dim_heads = self.head_dim,
            nb_features = 4 * self.head_dim,
            causal = False
        )

    def forward(self, x):
        batch_size = x.shape[0]
        query, key, value = [l(vector).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
                     for l, vector in zip((self.fc_query, self.fc_key, self.fc_value), (x, x, x))]
        
        out = self.attn_fn(query, key, value)
        out = out.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.embed_dim)
        return self.fc_out(out)