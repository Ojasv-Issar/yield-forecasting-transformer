#!/usr/bin/env python
"""
Attention modules for MMST-ViT.

This module implements various attention mechanisms:
- MultiModalAttention: Cross-attention for fusing visual and weather features
- SpatialAttention: Self-attention for spatial grid aggregation
- TemporalAttention: Self-attention with bias for temporal modeling
- Transformer wrappers for building deep attention networks
"""

import math
from inspect import isfunction

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat


def exists(val):
    """Check if a value is not None."""
    return val is not None


def uniq(arr):
    """Return unique elements from array."""
    return {el: True for el in arr}.keys()


def default(val, d):
    """Return val if it exists, otherwise return default d."""
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    """Return maximum negative value for tensor dtype."""
    return -torch.finfo(t.dtype).max


def init_(tensor):
    """Initialize tensor with uniform distribution scaled by dimension."""
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# =============================================================================
# Feed-Forward Networks
# =============================================================================

class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation."""
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward network with optional GLU activation."""
    
    def __init__(
        self,
        dim: int,
        dim_out: int = None,
        mult: int = 4,
        glu: bool = False,
        dropout: float = 0.
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreNorm(nn.Module):
    """Pre-normalization wrapper for attention/feed-forward layers."""
    
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


# =============================================================================
# Attention Mechanisms
# =============================================================================

class MultiModalAttention(nn.Module):
    """
    Cross-attention for multi-modal fusion.
    
    Computes attention between query (visual features) and context (weather data).
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # Compute attention weights
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class SpatialAttention(nn.Module):
    """
    Self-attention for spatial grid aggregation.
    
    Computes attention across spatial grids within each time step.
    """
    
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TemporalAttention(nn.Module):
    """
    Self-attention for temporal modeling with optional bias.
    
    Computes attention across time steps with optional weather context bias.
    """
    
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(bias):
            bias = self.to_qkv(bias).chunk(3, dim=-1)
            qb, kb, _ = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), bias)
            bias = einsum('b h i d, b h j d -> b h i j', qb, kb) * self.scale
            dots += bias

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


# =============================================================================
# Transformer Blocks
# =============================================================================

class MultiModalTransformer(nn.Module):
    """
    Transformer for multi-modal feature fusion.
    
    Stacks multiple layers of cross-attention and feed-forward networks
    for fusing visual features with weather context.
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        context_dim: int = 9,
        mult: int = 4,
        dropout: float = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiModalAttention(
                    dim, context_dim=context_dim, heads=heads,
                    dim_head=dim_head, dropout=dropout
                )),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return self.norm(x)


class SpatialTransformer(nn.Module):
    """
    Transformer for spatial grid aggregation.
    
    Stacks multiple layers of spatial self-attention and feed-forward networks
    for aggregating features across spatial grids.
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mult: int = 4,
        dropout: float = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class TemporalTransformer(nn.Module):
    """
    Transformer for temporal modeling.
    
    Stacks multiple layers of temporal self-attention (with optional bias)
    and feed-forward networks for modeling temporal dependencies.
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mult: int = 4,
        dropout: float = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TemporalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, bias=bias) + x
            x = ff(x) + x
        return self.norm(x)
