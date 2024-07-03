import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, zero_out=-1):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale #[B, heads, patch_num, patch_num]
        if zero_out != -1:
            cls_attn = attn[:,:,0,1:]
            topk_vals, topk_indices = torch.topk(cls_attn, k=zero_out, dim=-1)

            cls_attn = cls_attn.detach().cpu().numpy()
            import numpy as np
            np.put_along_axis(cls_attn, topk_indices.detach().cpu().numpy(), -np.inf, axis=-1)
            attn[:,:,0,1:] = torch.tensor(cls_attn).to(attn)       

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class LinearTransformer(nn.Module):
    """ Linear Transformer """
    def __init__(self, embed_dim=768, depth=12):
        super().__init__()
        self.attn_blocks = nn.ModuleList([Attention(dim=embed_dim, num_heads=1) for _ in range(depth)])
        #self.mlp = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        #x = self.mlp(x)
        for attn_block in self.attn_blocks:
            attn_x, _ = attn_block(x)
            x = x + attn_x
            #x = attn_x
        return x[:, -1, -1] #right bottom entry

    def get_all_selfattention(self, x, zero_out_attn=-1):
        attns = []
        for i, blk in enumerate(self.attn_blocks):
            x, attn_layer = blk(x, zero_out_attn)
            attns.append(attn_layer.detach().cpu())
        return attns
