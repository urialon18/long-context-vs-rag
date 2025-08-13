#!/usr/bin/env python3
"""Download LooGLE dataset as CSV files."""

import os
from pathlib import Path

from datasets import load_dataset
import pandas as pd


def download_loogle_dataset():
    """Download LooGLE dataset and save as CSV files."""
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("Downloading LooGLE dataset from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset("bigai-nlco/LooGLE", "longdep_qa")
    
    print(f"Dataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")
    
    # Save each split as a CSV file
    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split...")
        print(f"Number of examples: {len(split_data)}")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(split_data)
        
        # Save as CSV
        csv_path = data_dir / f"loogle_{split_name}.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"Saved {split_name} split to {csv_path}")
        print(f"File size: {csv_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Show some basic statistics
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data:")
        print(df.head(2).to_string())
        print("-" * 80)
    
    print(f"\n✅ All splits downloaded successfully to {data_dir}/")
    print("Files created:")
    for csv_file in data_dir.glob("loogle_*.csv"):
        print(f"  - {csv_file.name}")


if __name__ == "__main__":
    download_loogle_dataset()
