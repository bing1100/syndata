import torch
from torch import nn
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32)
    exponent = exponent / (half_dim - 1.0)

    emb = torch.exp(exponent).to(device=timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]

    return torch.cat([emb.sin(), emb.cos()], dim=-1)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn_mat = self.attend(dots)
        attn = self.dropout(attn_mat)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, temb_dim, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, dim, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(), 
            torch.nn.Linear(temb_dim, dim)
        )

    def forward(self, x, temb):
        for attn, ff1, ff2 in self.layers:
            v = attn(x)
            x = v + x 
            x = ff1(x) + x
            x = self.time_emb_proj(temb) + x
            x = ff2(x) + x
            
        return self.norm(x)

class SDT(nn.Module):
    def __init__(self, time_dim, num_classes, cond_size, y_dim, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()


        self.timestep_input_dim = time_dim
        self.time_embed_dim = self.timestep_input_dim * 4

        self.time_embedding = nn.Sequential(
        nn.Linear(self.timestep_input_dim, self.time_embed_dim), 
        nn.SiLU(),
        nn.Linear(self.time_embed_dim, self.time_embed_dim))

        num_patches = (cond_size // patch_size)
        patch_dim = patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p) -> b h p', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.out_dim = y_dim
        self.to_cls_y = nn.Sequential(
            nn.Linear(y_dim, dim),
            nn.SiLU(),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, dim)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(self.time_embed_dim, dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # Regression
        self.mlp_head = nn.Linear(dim, self.out_dim)

    def forward(self, cond, x_in, timesteps, y):
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps],
                                     dtype=torch.long,
                                     device=cond.device)
        timesteps = torch.flatten(timesteps)
        timesteps = timesteps.broadcast_to(cond.shape[0])

        t_emb = sinusoidal_embedding(timesteps, self.timestep_input_dim)
        t_emb = self.time_embedding(t_emb).unsqueeze(1)

        cond = self.dropout(cond)
        x = self.to_patch_embedding(cond)
        b, n, _ = x.shape
        
        # x += self.pos_embedding[:, :(n)]
        
        cls_x_in = self.to_cls_y(x_in).unsqueeze(1)
        
        cls_tokens = self.label_emb(y.squeeze())
        x = torch.cat((cls_x_in, cls_tokens, x), dim=1)
        
        x = self.dropout(x)

        x = self.transformer(x, t_emb)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)