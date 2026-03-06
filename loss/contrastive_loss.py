#!/usr/bin/env python
"""
Contrastive learning loss functions.

This module implements the NT-Xent (Normalized Temperature-scaled Cross Entropy)
loss used in SimCLR for self-supervised contrastive learning.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(nn.Module):
    """
    NT-Xent contrastive loss for SimCLR-style training.
    
    Maximizes agreement between differently augmented views of the same
    image while minimizing agreement between different images.
    
    Args:
        batch_size: Number of samples in each batch
        device: Device to run computations on
        temperature: Temperature parameter for scaling similarities
    """
    
    def __init__(
        self,
        batch_size: int,
        device: str,
        temperature: float = 0.5
    ):
        super().__init__()
        self.batch_size = batch_size
        device = torch.device(device)
        
        # Register buffers for temperature and negative mask
        self.register_buffer(
            "temperature", 
            torch.tensor(temperature).to(device)
        )
        self.register_buffer(
            "negatives_mask",
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(device)
        )

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            emb_i: Embeddings from first augmented view, shape (B, D)
            emb_j: Embeddings from second augmented view, shape (B, D)
            
        Returns:
            Scalar loss value
        """
        # Normalize embeddings
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        # Concatenate embeddings from both views
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )

        # Extract positive pairs (corresponding indices from both views)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        # Compute loss
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)

        return loss
