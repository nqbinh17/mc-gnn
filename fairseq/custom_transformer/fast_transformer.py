import torch, torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.encoder.embed_dim
        self.heads = cfg.encoder.attention_heads
        #assert self.embed_dim % self.heads
        self.head_dim = self.embed_dim // self.heads
        self.eps = 1e-6

        self.fc_query = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_key = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_value = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)

    def forward_kernel(self, x):
        return nn.functional.elu(x) + 1
    
    def forward(self, x, state):
        N, H, D = x.shape
        query, key, value = \
            [l(vector) for l, vector in zip((self.fc_query, self.fc_key, self.fc_value), (x, x, x))]

        kernel_key = self.forward_kernel(key)
        kernel_query = self.forward_kernel(query)

        if state is None:
            Zi = kernel_key
            Si = torch.einsum("nhd,nhm->nhdm", kernel_key, value)
        else:
            Si, Zi = state
            Zi = Zi + kernel_key
            Si = Si + torch.einsum("nhd,nhm->nhdm", kernel_key, value)

        Z = 1. / (torch.einsum("nhd,nhd->nh", kernel_query, Zi) + self.eps)
        V = torch.einsum("nhd,nhdm,nh->nhm", kernel_query, Si, Z)
        V = self.fc_out(V)
        return V, [Si, Zi]