#!/usr/bin/env python3
"""Verify the downloaded LooGLE dataset and show basic statistics."""

import pandas as pd
from pathlib import Path


def verify_loogle_data():
    """Verify the downloaded LooGLE dataset and show statistics."""
    
    data_dir = Path("data")
    csv_file = data_dir / "loogle_test.csv"
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    print(f"📊 Loading LooGLE dataset from {csv_file}")
    
    # Load the dataset
    df = pd.read_csv(csv_file)
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"📈 Dataset Statistics:")
    print(f"   - Total rows: {len(df):,}")
    print(f"   - Total columns: {len(df.columns)}")
    print(f"   - Columns: {list(df.columns)}")
    
    # Show basic statistics for text columns
    print(f"\n📝 Text Statistics:")
    print(f"   - Context length (chars):")
    print(f"     * Min: {df['context'].str.len().min():,}")
    print(f"     * Max: {df['context'].str.len().max():,}")
    print(f"     * Mean: {df['context'].str.len().mean():.0f}")
    print(f"     * Median: {df['context'].str.len().median():.0f}")
    
    print(f"   - Question length (chars):")
    print(f"     * Min: {df['question'].str.len().min():,}")
    print(f"     * Max: {df['question'].str.len().max():,}")
    print(f"     * Mean: {df['question'].str.len().mean():.0f}")
    
    print(f"   - Answer length (chars):")
    print(f"     * Min: {df['answer'].str.len().min():,}")
    print(f"     * Max: {df['answer'].str.len().max():,}")
    print(f"     * Mean: {df['answer'].str.len().mean():.0f}")
    
    # Show sample data
    print(f"\n📋 Sample Data (first 2 rows):")
    for i in range(min(2, len(df))):
        print(f"\n--- Row {i+1} ---")
        print(f"ID: {df.iloc[i]['id']}")
        print(f"Title: {df.iloc[i]['title']}")
        print(f"Question: {df.iloc[i]['question'][:100]}...")
        print(f"Answer: {df.iloc[i]['answer'][:100]}...")
        print(f"Context length: {len(df.iloc[i]['context']):,} characters")
    
    # Check for unique values
    print(f"\n🔍 Unique Values:")
    print(f"   - Unique doc_ids: {df['doc_id'].nunique():,}")
    print(f"   - Unique titles: {df['title'].nunique():,}")
    
    print(f"\n✅ Dataset verification complete!")


if __name__ == "__main__":
    verify_loogle_data()
