# General
import math

# Torch
import torch
import torch.nn as nn

# Wordle



class MHA(nn.Module):
    def __init__(self, dim, bias=False, activation=nn.SiLU(), dropout=0.0):
        # Setup
        super().__init__()
        self.dim = dim
        self.bias = bias
        self.act = activation
        self.dropout = dropout

        # Modules
        self.qkv = nn.Linear(dim, 3*dim, bias=bias)

    def forward(self, x):
        d = math.sqrt(x.shape[-1])
        q, k, v = self.qkv(x).chunk(3, dim=-1)                  # [B, 5, D] each
        a = torch.einsum("b i d, b j d -> b i j", q, k) / d     # [B, 5, 5]
        v = torch.einsum("b i j, b j d -> b i d", a, v)         # [B, 5, D]
        return v

class FF(nn.Module):
    def __init__(self, dim, activation=nn.SiLU(), dropout=0.0):
        # Setup
        super().__init__()
        self.dim = dim

        # Modules
        self.ff1 = nn.Linear(dim, 4*dim)
        self.act = activation
        self.dropout = nn.Dropout(dropout)
        self.ff2 = nn.Linear(4*dim, dim)

    def forward(self, x):
        x = self.ff1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.ff2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, bias=False, activation=nn.SiLU(), dropout=0.0):
        # Setup
        super().__init__()
        self.dim = dim
        self.bias = bias
        self.act = activation
        self.dropout = dropout

        # Modules
        self.norm1 = nn.LayerNorm(dim)
        self.mha = MHA(dim, bias=bias, activation=activation, dropout=dropout)
        self.ff = FF(dim, activation=activation, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class MLPBlock(nn.Module):
    def __init__(self, dim, activation=nn.SiLU(), dropout=0.0):
        super().__init__()
        self.res = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            activation,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.res(x)
