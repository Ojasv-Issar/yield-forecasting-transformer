#!/usr/bin/env python
"""
Test script to verify MMST-ViT model can be instantiated and run.
This doesn't require the full dataset - it uses random tensors.
"""

import torch
from models_pvt_simclr import PVTSimCLR
from models_mmst_vit import MMST_ViT


def test_pvt_simclr():
    """Test PVT SimCLR backbone model"""
    print("=" * 50)
    print("Testing PVT SimCLR model...")
    print("=" * 50)
    
    # Create model
    model = PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9)
    print(f"Model created: PVTSimCLR with pvt_tiny backbone")
    
    # Create random input
    # x: input image (B, C, H, W)
    x = torch.randn(2, 3, 224, 224)
    # context: weather/meteorological data (B, N, D) where N=number of time steps, D=features
    context = torch.randn(2, 28, 9)
    
    print(f"Input shape: {x.shape}")
    print(f"Context shape: {context.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x, context=context)
    
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0, :5].tolist()}")
    print("PVT SimCLR test PASSED!\n")


def test_mmst_vit():
    """Test full MMST-ViT model"""
    print("=" * 50)
    print("Testing MMST-ViT model...")
    print("=" * 50)
    
    # Create PVT backbone
    pvt = PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9)
    
    # Create MMST-ViT model
    # Note: num_grid should be >= actual_grids + 1 (for cls token)
    model = MMST_ViT(
        out_dim=4,  # 4 output features (could be yield predictions)
        pvt_backbone=pvt,
        dim=512,
        num_grid=64,  # maximum number of spatial grids (must be >= actual grids used)
        num_short_term_seq=6,  # 6 short-term time steps (e.g., 6 months)
        num_long_term_seq=12,  # 12 long-term time steps
        num_year=5,  # 5 years of historical data
        batch_size=32
    )
    print(f"Model created: MMST-ViT")
    
    # Create random inputs
    # x: satellite imagery (B, T, G, C, H, W)
    # B=batch, T=time steps, G=grids, C=channels, H=height, W=width
    x = torch.randn(1, 6, 10, 3, 224, 224)
    
    # ys: short-term weather data (B, T, G, N1, D)
    ys = torch.randn(1, 6, 10, 28, 9)
    
    # yl: long-term climate data (B, Y, M, D)
    # Y=years, M=months
    yl = torch.randn(1, 5, 12, 9)
    
    print(f"Input x shape (satellite): {x.shape}")
    print(f"Input ys shape (short-term weather): {ys.shape}")
    print(f"Input yl shape (long-term climate): {yl.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x, ys=ys, yl=yl)
    
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    print("MMST-ViT test PASSED!\n")


def main():
    print("\n" + "=" * 60)
    print("MMST-ViT Project Test Suite")
    print("Testing model instantiation and forward pass...")
    print("=" * 60 + "\n")
    
    # Test dependencies
    print("Verifying dependencies...")
    import torch
    import timm
    import einops
    import numpy as np
    print(f"  PyTorch: {torch.__version__}")
    print(f"  timm: {timm.__version__}")
    print(f"  einops: {einops.__version__}")
    print(f"  numpy: {np.__version__}")
    print("Dependencies OK!\n")
    
    # Run tests
    test_pvt_simclr()
    test_mmst_vit()
    
    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
    print("\nThe project is set up correctly.")
    print("To train the model, you need to:")
    print("1. Download the Tiny CropNet dataset from HuggingFace:")
    print("   https://huggingface.co/datasets/fudong03/Tiny-CropNet")
    print("2. Extract the dataset and update the --root_dir argument")
    print("3. Run pre-training:")
    print("   python main_pretrain_mmst_vit.py --root_dir <path-to-dataset> --device cpu")
    print("4. Run fine-tuning:")
    print("   python main_finetune_mmst_vit.py --root_dir <path-to-dataset> --device cpu")


if __name__ == "__main__":
    main()
