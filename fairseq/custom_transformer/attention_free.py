import torch
import torch.nn as nn

class AFTFullAttention(nn.Module):
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

    def forward(self, x):
        # Extract some shapes
        batch_size, _, _ = x.shape
        query, key, value = [l(vector).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
                     for l, vector in zip((self.fc_query, self.fc_key, self.fc_value), (x, x, x))]

        Q = torch.sigmoid(query)
        K = key
        K = torch.softmax(K, dim=-1) 
        V = Q * (K * value).sum(dim=1, keepdim=True)
        V = V.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.embed_dim)
        return self.fc_out(V)