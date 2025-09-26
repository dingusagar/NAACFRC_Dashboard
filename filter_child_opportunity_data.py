#!/usr/bin/env python3
"""
Script to filter the Child Opportunity Index dataset.
Filters for Georgia data and keeps only the first 12 columns.
"""

import pandas as pd
import os

def filter_child_opportunity_data():
    """
    Filter the Child Opportunity Index dataset for Georgia and keep first 12 columns.
    """
    # File paths
    input_file = "data/child_opportunity_index.csv"
    output_file = "data/child_opportunity_index_georgia_filtered.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    print(f"Reading data from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns in original dataset: {list(df.columns)}")
    
    # Filter for Georgia
    georgia_df = df[df['state_name'] == 'Georgia'].copy()
    
    print(f"Georgia dataset shape after state filter: {georgia_df.shape}")
    
    # Get the first 12 columns
    first_12_columns = df.columns[:12].tolist()
    print(f"First 12 columns: {first_12_columns}")
    
    # Keep only the first 12 columns
    filtered_df = georgia_df[first_12_columns].copy()
    
    print(f"Final filtered dataset shape: {filtered_df.shape}")
    
    # Save the filtered dataset
    filtered_df.to_csv(output_file, index=False)
    
    print(f"Filtered dataset saved to {output_file}")
    
    # Display some summary statistics
    print("\nSummary of the filtered dataset:")
    print(f"Number of records: {len(filtered_df)}")
    print(f"Unique counties: {filtered_df['county_name'].nunique()}")
    print(f"Years covered: {sorted(filtered_df['year'].unique())}")
    
    # Display first few rows
    print("\nFirst 5 rows of the filtered dataset:")
    print(filtered_df.head())

if __name__ == "__main__":
    filter_child_opportunity_data()