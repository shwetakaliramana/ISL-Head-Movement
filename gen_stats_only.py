#!/usr/bin/env python3
"""
Quick script to generate normaliser_stats.npz from the dataset.
"""
import sys
from src.ml.feature_engineering import DatasetBuilder

def main():
    print("[INFO] Generating normaliser stats from dataset...")
    builder = DatasetBuilder()
    X, y = builder.fit_transform('data/raw/synthetic_dataset.csv')
    
    # Save stats
    builder.save_stats('models/normaliser_stats.npz')
    print(f"[DONE] Saved normaliser stats to models/normaliser_stats.npz")
    print(f"       Shape: {X.shape}, Classes: {len(set(y))}")

if __name__ == '__main__':
    main()
