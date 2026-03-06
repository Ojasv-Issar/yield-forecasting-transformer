#!/usr/bin/env python
"""
USDA crop statistics data loader.

This module provides a PyTorch Dataset for loading USDA crop production
and yield statistics for crop yield forecasting.
"""

import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


class USDADataset(Dataset):
    """
    Dataset for loading USDA crop statistics.
    
    Loads annual crop production and yield data for specified counties.
    Output values are log-transformed for numerical stability during training.
    
    Args:
        root_dir: Root directory containing the USDA data files
        json_file: Path to JSON file containing data index
        crop_type: Type of crop ('Soybeans', 'Cotton', etc.)
        
    Returns:
        x: Log-transformed tensor of [production, yield]
        fips_code: County FIPS code
        year: Data year
    """

    # Column mappings by crop type
    CROP_COLUMNS = {
        'Cotton': ['PRODUCTION, MEASURED IN 480 LB BALES', 'YIELD, MEASURED IN BU / ACRE'],
        'default': ['PRODUCTION, MEASURED IN BU', 'YIELD, MEASURED IN BU / ACRE']
    }

    def __init__(self, root_dir: str, json_file: str, crop_type: str = "Soybeans"):
        self.crop_type = crop_type
        self.select_cols = self.CROP_COLUMNS.get(crop_type, self.CROP_COLUMNS['default'])

        self.fips_codes = []
        self.years = []
        self.state_ansi = []
        self.county_ansi = []
        self.file_paths = []

        with open(json_file) as f:
            data = json.load(f)
            
        for obj in data:
            self.fips_codes.append(obj["FIPS"])
            self.years.append(obj["year"])
            self.state_ansi.append(obj["state_ansi"])
            self.county_ansi.append(obj["county_ansi"])
            self.file_paths.append(os.path.join(root_dir, obj["data"]["USDA"]))

    def __len__(self) -> int:
        return len(self.fips_codes)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, int]:
        fips_code = self.fips_codes[index]
        year = self.years[index]
        state_ansi = self.state_ansi[index]
        county_ansi = self.county_ansi[index]
        file_path = self.file_paths[index]

        df = pd.read_csv(file_path)

        # Convert state_ansi and county_ansi to string with leading zeros
        df['state_ansi'] = df['state_ansi'].astype(str).str.zfill(2)
        df['county_ansi'] = df['county_ansi'].astype(str).str.zfill(3)

        # Filter by county
        df = df[(df["state_ansi"] == state_ansi) & (df["county_ansi"] == county_ansi)]
        df = df[self.select_cols]

        # Convert to tensor and apply log transform
        x = torch.from_numpy(df.values)
        x = x.to(torch.float32)
        x = torch.log(torch.flatten(x, start_dim=0))

        return x, fips_code, year


if __name__ == '__main__':
    """Test the USDADataset loader."""
    root_dir = "/mnt/data/Tiny CropNet"
    data_file = "./../data/soybean_val.json"

    dataset = USDADataset(root_dir, data_file)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for x, fips, year in loader:
        print(f"FIPS: {fips}, Year: {year}, Shape: {x.shape}")
