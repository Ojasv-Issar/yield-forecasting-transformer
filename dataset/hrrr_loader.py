#!/usr/bin/env python
"""
HRRR (High-Resolution Rapid Refresh) weather data loader.

This module provides a PyTorch Dataset for loading HRRR meteorological data
for crop yield forecasting. Includes both short-term (daily) and long-term
(monthly) weather variables.
"""

import concurrent.futures
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


class HRRRDataset(Dataset):
    """
    Dataset for loading HRRR weather data.
    
    Loads short-term (daily) and long-term (monthly) meteorological variables
    for specified counties. Variables include temperature, precipitation,
    humidity, wind, radiation, and vapor pressure deficit.
    
    Args:
        root_dir: Root directory containing the HRRR data files
        json_file: Path to JSON file containing data index
        num_workers: Number of threads for parallel file reading
        
    Returns:
        x_short: Short-term weather tensor of shape (T, G, N, D)
                 T=months, G=grids, N=days (28), D=features (9)
        x_long: Long-term weather tensor of shape (Y, M, D)
                Y=years, M=months, D=features
        fips_code: County FIPS code
        year: Data year
    """

    # Weather variables to extract
    WEATHER_FEATURES = [
        'Avg Temperature (K)',
        'Max Temperature (K)', 
        'Min Temperature (K)',
        'Precipitation (kg m**-2)',
        'Relative Humidity (%)',
        'Wind Gust (m s**-1)',
        'Wind Speed (m s**-1)',
        'Downward Shortwave Radiation Flux (W m**-2)',
        'Vapor Pressure Deficit (kPa)'
    ]

    def __init__(self, root_dir: str, json_file: str, num_workers: int = 4):
        # Consider the first 28 days in each month
        self.day_range = list(range(1, 29))
        self.select_cols = self.WEATHER_FEATURES
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        self.fips_codes = []
        self.years = []
        self.short_term_file_path = []
        self.long_term_file_path = []

        with open(json_file) as f:
            data = json.load(f)

        for obj in data:
            self.fips_codes.append(obj["FIPS"])
            self.years.append(obj["year"])

            # Short-term file paths
            short_term = [
                os.path.join(root_dir, fp) 
                for fp in obj["data"]["HRRR"]["short_term"]
            ]
            self.short_term_file_path.append(short_term)

            # Long-term file paths (nested list)
            long_term = [
                [os.path.join(root_dir, fp) for fp in file_paths]
                for file_paths in obj["data"]["HRRR"]["long_term"]
            ]
            self.long_term_file_path.append(long_term)

    def __len__(self) -> int:
        return len(self.fips_codes)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        fips_code = self.fips_codes[index]
        year = self.years[index]

        # Load short-term and long-term weather data
        short_term_paths = self.short_term_file_path[index]
        x_short = self._get_short_term_data(fips_code, short_term_paths)

        long_term_paths = self.long_term_file_path[index]
        x_long = self._get_long_term_data(fips_code, long_term_paths)

        # Convert to float32
        x_short = x_short.to(torch.float32)
        x_long = x_long.to(torch.float32)

        return x_short, x_long, fips_code, year

    def _get_short_term_data(
        self, 
        fips_code: str, 
        file_paths: List[str]
    ) -> torch.Tensor:
        """Load and process short-term (daily) weather data."""
        df_list = [self._read_csv_file(fp) for fp in file_paths]
        df = pd.concat(df_list, ignore_index=True)

        # Filter by county and daily data
        df["FIPS Code"] = df["FIPS Code"].astype(str)
        df = df[(df["FIPS Code"] == fips_code) & (df["Daily/Monthly"] == "Daily")]
        df.columns = df.columns.str.strip()

        temporal_list = []
        for month, df_month in df.groupby(['Month']):
            time_series = []
            for grid, df_grid in df_month.groupby(['Grid Index']):
                df_grid = df_grid.sort_values(by=['Day'], ascending=True, na_position='first')
                df_grid = df_grid[df_grid.Day.isin(self.day_range)]
                df_grid = df_grid[self.select_cols]
                time_series.append(torch.from_numpy(df_grid.values))
            temporal_list.append(torch.stack(time_series))

        return torch.stack(temporal_list)

    def _get_long_term_data(
        self, 
        fips_code: str, 
        temporal_file_paths: List[List[str]]
    ) -> torch.Tensor:
        """Load and process long-term (monthly) weather data."""
        temporal_list = []

        for file_paths in temporal_file_paths:
            # Parallel file reading
            futures = [
                self.executor.submit(self._read_csv_file, fp) 
                for fp in file_paths
            ]
            concurrent.futures.wait(futures)
            dfs = [f.result() for f in futures]

            df = pd.concat(dfs, ignore_index=True)

            # Filter by county and monthly data
            df["FIPS Code"] = df["FIPS Code"].astype(str)
            df = df[(df["FIPS Code"] == fips_code) & (df["Daily/Monthly"] == "Monthly")]
            df.columns = df.columns.str.strip()

            month_list = []
            for month, df_month in df.groupby(['Month']):
                df_month = df_month[self.select_cols]
                val = torch.from_numpy(df_month.values)
                val = torch.flatten(val, start_dim=0)
                month_list.append(val)

            temporal_list.append(torch.stack(month_list))

        return torch.stack(temporal_list)

    @lru_cache(maxsize=128)
    def _read_csv_file(self, file_path: str) -> pd.DataFrame:
        """Read CSV file with caching."""
        return pd.read_csv(file_path)


if __name__ == '__main__':
    """Test the HRRRDataset loader."""
    root_dir = "/mnt/data/Tiny CropNet"
    data_file = "./../data/soybean_train.json"
    
    dataset = HRRRDataset(root_dir, data_file)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for x_short, x_long, fips, year in loader:
        print(f"FIPS: {fips}, Year: {year}")
        print(f"  Short-term shape: {x_short.shape}")
        print(f"  Long-term shape: {x_long.shape}")
        break
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # Record start time
    start_time = time.time()
    for xs, xl, f, y in train_loader:
        print("fips: {}, year: {}, short shape: {}".format(f, y, xs.shape))
        print("fips: {}, year: {}, long shape: {}".format(f, y, xl.shape))

        # Record end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        print(f"Time Elapsed: {elapsed_time:.6f} seconds")

        start_time = time.time()
