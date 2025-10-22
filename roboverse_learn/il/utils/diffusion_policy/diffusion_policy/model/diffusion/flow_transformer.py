import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from itertools import repeat
import collections.abc

from diffusion_policy.model.diffusion.positional_embedding import RotaryPosEmb, SinusoidalPosEmb


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer (timm.layers)"""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        max_seq_len=16,
        qk_norm=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # self.attn_norm = self.head_dim ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryPosEmb(dim, max_seq_len=max_seq_len)

    def forward(self, x, mask=None):
        B, S, C = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, S, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)
        if self.training:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop)
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, S, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AdaLNBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        dropout=0.,
    ):
        super().__init__()
        # self.norm1 = RMSNorm(dim)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout
        )

        def approx_gelu(): return nn.GELU(approximate="tanh")

        # self.norm2 = RMSNorm(dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=0
        )

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6*dim)
        )
        self.dim = dim

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.ada_ln[-1].weight, 0)
        nn.init.constant_(self.ada_ln[-1].bias, 0)

    def forward(self, x, t, c):
        B = x.shape[0]
        features = self.ada_ln(nn.SiLU()(t+c)).view(B, 6, 1, self.dim).unbind(1)
        gamma1, gamma2, scale1, scale2, shift1, shift2 = features

        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1.mul(scale1.add(1)).add_(shift1)
        x = x + self.attn(x_norm1).mul_(gamma1)

        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2.mul(scale2.add(1)).add_(shift2)
        x = x + self.mlp(x_norm2).mul_(gamma2)

        return x


class FlowTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        condition_dim,
        hidden_dim,
        output_dim,
        num_layers,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        time_embed_dim=256,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )
        self.cond_embed = nn.Linear(condition_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            AdaLNBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        # Basic initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)

    def forward(self, x, t, global_cond, local_cond=None):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(global_cond)

        for block in self.transformer_blocks:
            x = block(x, t, c)

        x = self.norm(x)
        x = self.out_proj(x)
        return x
