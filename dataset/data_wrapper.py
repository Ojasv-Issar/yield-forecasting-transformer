#!/usr/bin/env python
"""
Data augmentation wrappers for training and validation.

This module provides data transformation pipelines including SimCLR-style
augmentations for contrastive learning pre-training.
"""

import torch
import torchvision.transforms as transforms
from einops import rearrange
from sklearn import preprocessing

from dataset.sentinel_loader import SentinelDataset


class DataWrapper:
    """
    Data augmentation wrapper for satellite imagery.
    
    Applies SimCLR-style augmentations for training or simple transforms
    for validation/inference.
    
    Args:
        img_size: Target image size after transforms
        s: Color jitter strength multiplier
        kernel_size: Gaussian blur kernel size
        train: Whether to use training augmentations
    """
    
    def __init__(
        self,
        img_size: int = 224,
        s: float = 1.0,
        kernel_size: int = 9,
        train: bool = True
    ):
        self.img_size = img_size
        self.s = s
        self.kernel_size = kernel_size

        if train:
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_val_transform()

    def __call__(self, x: torch.Tensor):
        """
        Apply augmentations to input tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            xi, xj: Two augmented views of the input
        """
        x = x.to(torch.float32)
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj

    def _get_train_transform(self):
        """Get SimCLR training augmentation pipeline."""
        color_jitter = transforms.ColorJitter(
            0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s
        )
        return transforms.Compose([
            transforms.RandomResizedCrop(size=self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=self.kernel_size),
            transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]),
        ])

    def _get_val_transform(self):
        """Get validation/inference transform pipeline."""
        return transforms.Compose([
            transforms.CenterCrop(size=self.img_size),
            transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]),
        ])


class ScalarNorm:
    """
    Standard scalar normalization using sklearn's StandardScaler.
    
    Normalizes features to zero mean and unit variance.
    """
    
    def __init__(self):
        self.norm = preprocessing.StandardScaler()

    def __call__(self, x, reverse: bool = False) -> torch.Tensor:
        """
        Apply or reverse normalization.
        
        Args:
            x: Input array
            reverse: If True, apply inverse transform
            
        Returns:
            Normalized/denormalized tensor
        """
        if not reverse:
            x = self.norm.fit_transform(x)
        else:
            x = self.norm.inverse_transform(x)

        x = torch.from_numpy(x)
        x = x.to(torch.float32)
        return x


if __name__ == '__main__':
    """Test the DataWrapper."""
    root_dir = "/mnt/data/Crop"
    data_file = "./../data/soybean_train.json"
    
    dataset = SentinelDataset(root_dir, data_file)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    it = iter(loader)
    x, fips, year = next(it)
    x = x[:, :, :22, :, :, :]

    x = rearrange(x, 'b t g h w c -> (b t g) c h w')

    wrapper = DataWrapper()
    xi, xj = wrapper(x)
    
    print(f"Augmented view 1 shape: {xi.shape}")
    print(f"Augmented view 2 shape: {xj.shape}")
