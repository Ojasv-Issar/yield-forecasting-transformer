#!/usr/bin/env python
"""
MMST-ViT: Multi-Modal Spatio-Temporal Vision Transformer for crop yield forecasting.

This module implements the main MMST-ViT architecture that combines:
- PVT-SimCLR backbone for processing satellite imagery
- Spatial transformer for aggregating features across spatial grids
- Temporal transformer for modeling temporal dependencies
- Multi-modal fusion for incorporating weather data
"""

import torch
from torch import nn
from einops import rearrange, repeat

from attention import SpatialTransformer, TemporalTransformer
from models_pvt_simclr import PVTSimCLR


class MMSTViT(nn.Module):
    """
    Multi-Modal Spatio-Temporal Vision Transformer for crop yield prediction.
    
    This model processes satellite imagery along with short-term and long-term
    weather data to predict crop yields at the county level.
    
    Args:
        out_dim: Output dimension (number of predictions, e.g., production and yield)
        num_grid: Maximum number of spatial grids per county
        num_short_term_seq: Number of short-term time steps (months in growing season)
        num_long_term_seq: Number of long-term time steps (months per year)
        num_year: Number of years of historical data
        pvt_backbone: Pre-trained PVT-SimCLR backbone model
        context_dim: Dimension of weather context features
        dim: Model embedding dimension
        batch_size: Batch size for processing grids
        depth: Number of transformer layers
        heads: Number of attention heads
        pool: Pooling type ('cls' for class token, 'mean' for mean pooling)
        dim_head: Dimension of each attention head
        dropout: Dropout rate
        emb_dropout: Embedding dropout rate
        scale_dim: Feed-forward network scale factor
    """
    
    def __init__(
        self,
        out_dim: int = 2,
        num_grid: int = 64,
        num_short_term_seq: int = 6,
        num_long_term_seq: int = 12,
        num_year: int = 5,
        pvt_backbone: nn.Module = None,
        context_dim: int = 9,
        dim: int = 192,
        batch_size: int = 64,
        depth: int = 4,
        heads: int = 3,
        pool: str = 'cls',
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
        scale_dim: int = 4,
    ):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.batch_size = batch_size
        self.pvt_backbone = pvt_backbone

        self.proj_context = nn.Linear(num_year * num_long_term_seq * context_dim, num_short_term_seq * dim)
        # self.proj_context = nn.Linear(num_year * num_long_term_seq * context_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_short_term_seq, num_grid, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = SpatialTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = TemporalTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.norm1 = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    def forward_features(self, x: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """
        Extract features from satellite imagery using the PVT backbone.
        
        Args:
            x: Satellite imagery tensor of shape (B, T, G, C, H, W)
            ys: Short-term weather data of shape (B, T, G, N, D)
            
        Returns:
            Feature tensor of shape (B*T*G, dim)
        """
        x = rearrange(x, 'b t g c h w -> (b t g) c h w')
        ys = rearrange(ys, 'b t g n d -> (b t g) n d')

        # Process in batches to prevent OOM for large grid counts
        B = x.shape[0]
        n = B // self.batch_size if B % self.batch_size == 0 else B // self.batch_size + 1

        x_hat = torch.empty(0).to(x.device)
        for i in range(n):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            x_tmp = x[start:end]
            ys_tmp = ys[start:end]

            x_hat_tmp = self.pvt_backbone(x_tmp, context=ys_tmp)
            x_hat = torch.cat([x_hat, x_hat_tmp], dim=0)

        return x_hat

    def forward(
        self,
        x: torch.Tensor,
        ys: torch.Tensor = None,
        yl: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the MMST-ViT model.
        
        Args:
            x: Satellite imagery of shape (B, T, G, C, H, W)
               B=batch, T=time steps, G=grids, C=channels, H=height, W=width
            ys: Short-term weather data of shape (B, T, G, N, D)
               N=days (28), D=features (9)
            yl: Long-term climate data of shape (B, Y, M, D)
               Y=years, M=months per year
               
        Returns:
            Predictions of shape (B, out_dim)
        """
        b, t, g, _, _, _ = x.shape
        x = self.forward_features(x, ys)
        x = rearrange(x, '(b t g) d -> b t g d', b=b, t=t, g=g)

        # Add spatial class tokens
        cls_space_tokens = repeat(self.space_token, '() g d -> b t g d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(g + 1)]
        x = self.dropout(x)

        # Spatial transformer
        x = rearrange(x, 'b t g d -> (b t) g d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        # Add temporal class tokens
        cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b t d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        # Project long-term weather context
        yl = rearrange(yl, 'b y m d -> b (y m d)')
        yl = self.proj_context(yl)
        yl = rearrange(yl, 'b (t d) -> b t d', t=t)
        # yl = repeat(yl, '() d -> b t d', b=b, t=t)

        # Temporal transformer with weather context bias
        yl = torch.cat((cls_temporal_tokens, yl), dim=1)
        yl = self.norm1(yl)

        x = self.temporal_transformer(x, yl)

        # Pool and predict
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)


if __name__ == "__main__":
    """Test model instantiation and forward pass."""
    
    # Satellite imagery: (Batch, Time, Grids, Channels, Height, Width)
    x = torch.randn((1, 6, 10, 3, 224, 224))
    
    # Short-term weather: (Batch, Time, Grids, Days, Features)
    ys = torch.randn((1, 6, 10, 28, 9))
    
    # Long-term climate: (Batch, Years, Months, Features)
    yl = torch.randn((1, 5, 12, 9))

    # Create model
    pvt = PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9)
    model = MMSTViT(out_dim=4, pvt_backbone=pvt, dim=512)

    # Forward pass
    z = model(x, ys=ys, yl=yl)
    print(f"Output: {z}")
    print(f"Output shape: {z.shape}")
