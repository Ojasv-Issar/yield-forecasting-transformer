#!/usr/bin/env python
"""
Sentinel-2 satellite imagery data loader.

This module provides a PyTorch Dataset for loading Sentinel-2 satellite imagery
stored in HDF5 files for crop yield forecasting.
"""

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


class SentinelDataset(Dataset):
    """
    Dataset for loading Sentinel-2 satellite imagery.
    
    Loads monthly satellite imagery for specified counties from HDF5 files.
    Each sample contains imagery across the growing season for a single county-year.
    
    Args:
        root_dir: Root directory containing the Sentinel-2 data files
        json_file: Path to JSON file containing data index
        
    Returns:
        x: Satellite imagery tensor of shape (T, G, H, W, C)
           T=time steps, G=grids, H=height, W=width, C=channels
        fips_code: County FIPS code
        year: Data year
    """

    def __init__(self, root_dir: str, json_file: str):
        self.fips_codes = []
        self.years = []
        self.file_paths = []

        with open(json_file) as f:
            data = json.load(f)
            
        for obj in data:
            self.fips_codes.append(obj["FIPS"])
            self.years.append(obj["year"])

            tmp_path = []
            relative_path_list = obj["data"]["sentinel"]
            for relative_path in relative_path_list:
                tmp_path.append(os.path.join(root_dir, relative_path))
            self.file_paths.append(tmp_path)

    def __len__(self) -> int:
        return len(self.fips_codes)

    def __getitem__(self, index: int):
        fips_code, year = self.fips_codes[index], self.years[index]
        file_paths = self.file_paths[index]

        temporal_list = []

        for file_path in file_paths:
            with h5py.File(file_path, 'r') as hf:
                groups = hf[fips_code]
                for i, d in enumerate(groups.keys()):
                    # Only consider the 1st day of each month
                    # Note: HDF5 contains 1st and 15th of each month (e.g., "04-01" and "04-15")
                    if i % 2 == 0:
                        grids = groups[d]["data"]
                        grids = np.asarray(grids)
                        temporal_list.append(torch.from_numpy(grids))

        x = torch.stack(temporal_list)

        return x, fips_code, year


if __name__ == '__main__':
    """Test the SentinelDataset loader."""
    root_dir = "/mnt/data/Tiny CropNet"
    data_file = "./../data/soybean_val.json"
    
    dataset = SentinelDataset(root_dir, data_file)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    max_grids = 0
    for x, fips, year in loader:
        print(f"FIPS: {fips}, Year: {year}, Shape: {x.shape}")
        max_grids = max(max_grids, x.shape[2])

    print(f"Maximum number of grids: {max_grids}")
