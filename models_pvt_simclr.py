#!/usr/bin/env python
"""
PVT-SimCLR: Pyramid Vision Transformer with SimCLR projection head.

This module implements the PVT backbone with a multi-modal transformer
for integrating visual features with weather context data.
"""

import torch
from torch import nn

import models_pvt
from attention import MultiModalTransformer


class PVTSimCLR(nn.Module):
    """
    Pyramid Vision Transformer backbone with SimCLR-style projection and
    multi-modal attention for weather context fusion.
    
    Args:
        base_model: Name of PVT variant ('pvt_tiny', 'pvt_small', 'pvt_medium', 'pvt_large')
        out_dim: Output embedding dimension
        context_dim: Dimension of weather context features
        num_head: Number of attention heads in multi-modal transformer
        mm_depth: Depth of multi-modal transformer
        dropout: Dropout rate
        pretrained: Whether to use pretrained PVT weights
        gated_ff: Whether to use gated feed-forward (unused, for compatibility)
    """

    def __init__(
        self,
        base_model: str,
        out_dim: int = 512,
        context_dim: int = 9,
        num_head: int = 8,
        mm_depth: int = 2,
        dropout: float = 0.,
        pretrained: bool = True,
        gated_ff: bool = True
    ):
        super().__init__()

        # Initialize PVT backbone
        self.backbone = models_pvt.__dict__[base_model](pretrained=pretrained)
        num_ftrs = self.backbone.head.in_features

        # Projection layers
        self.proj = nn.Linear(num_ftrs, out_dim)
        self.proj_context = nn.Linear(context_dim, out_dim)

        # Multi-modal transformer for context fusion
        dim_head = out_dim // num_head
        self.mm_transformer = MultiModalTransformer(
            out_dim, mm_depth, num_head, dim_head,
            context_dim=out_dim, dropout=dropout
        )

        self.norm1 = nn.LayerNorm(context_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (B, C, H, W)
            context: Weather context of shape (B, N, D) where N=time steps, D=features
            
        Returns:
            Feature embedding of shape (B, out_dim)
        """
        # Extract visual features
        h = self.backbone.forward_features(x)  # shape = (B, N, D)
        h = h.squeeze()

        # Project to target dimension
        x = self.proj(h)
        context = self.proj_context(self.norm1(context))

        # Multi-modal attention fusion
        x = self.mm_transformer(x, context=context)

        # Return the classification token
        return x[:, 0]
