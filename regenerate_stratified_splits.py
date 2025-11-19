#!/usr/bin/env python3
"""
Regenerate train/val/test splits with proper stratification
This fixes the critical data distribution mismatch issue.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

# Configuration
BASE_DIR = "darpa_tc"
PROCESSED_DIR = f"{BASE_DIR}/processed"
SPLITS_DIR = f"{BASE_DIR}/splits"
METADATA_DIR = f"{BASE_DIR}/metadata"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
RANDOM_STATE = 42

def main():
    print("=" * 80)
    print("REGENERATING STRATIFIED SPLITS")
    print("=" * 80)
    print()

    # Create metadata directory if it doesn't exist
    Path(METADATA_DIR).mkdir(parents=True, exist_ok=True)

    # Load all processed chunks
    processed_path = Path(PROCESSED_DIR)
    parquet_files = sorted(processed_path.glob('*.parquet'))

    if not parquet_files:
        print(f"❌ No processed files found in {PROCESSED_DIR}")
        return

    print(f"✓ Found {len(parquet_files)} chunk files")

    # Load and concatenate all chunks
    print("📥 Loading chunks...")
    dfs = []
    for i, f in enumerate(parquet_files):
        df = pd.read_parquet(f)
        dfs.append(df)
        if (i + 1) % 5 == 0:
            print(f"   Loaded {i+1}/{len(parquet_files)} chunks...")

    print("🔗 Concatenating data...")
    full_df = pd.concat(dfs, ignore_index=True)

    print(f"✓ Total records: {len(full_df):,}")
    print()

    # Check current distribution
    total_threats = full_df['is_suspicious'].sum()
    threat_pct = total_threats / len(full_df) * 100
    print(f"Overall threat distribution: {total_threats:,}/{len(full_df):,} = {threat_pct:.2f}%")
    print()

    # CRITICAL FIX: Use stratified splitting
    print("🔄 Creating STRATIFIED splits...")
    print("   This ensures all splits have the SAME threat distribution!")
    print()

    # Extract labels for stratification
    labels = full_df['is_suspicious'].values

    # First split: 70% train, 30% temp (val + test)
    print("   Step 1: Splitting into train (70%) and temp (30%)...")
    train_df, temp_df = train_test_split(
        full_df,
        test_size=(1 - TRAIN_RATIO),
        stratify=labels,  # ← CRITICAL: Ensures same threat ratio
        random_state=RANDOM_STATE
    )

    # Second split: Split temp into val (15%) and test (15%)
    print("   Step 2: Splitting temp into val (15%) and test (15%)...")
    temp_labels = temp_df['is_suspicious'].values
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # 50% of temp = 15% of total
        stratify=temp_labels,  # ← CRITICAL: Ensures same threat ratio
        random_state=RANDOM_STATE
    )

    print("✓ Stratified splitting complete!")
    print()

    # Calculate metrics
    train_threats = train_df['is_suspicious'].sum()
    val_threats = val_df['is_suspicious'].sum()
    test_threats = test_df['is_suspicious'].sum()

    train_threat_pct = train_threats / len(train_df) * 100
    val_threat_pct = val_threats / len(val_df) * 100
    test_threat_pct = test_threats / len(test_df) * 100

    # Print summary BEFORE saving
    print("=" * 80)
    print("📊 SPLIT STATISTICS (STRATIFIED)")
    print("=" * 80)
    print()
    print(f"Dataset Statistics:")
    print(f"  Total Records:     {len(full_df):,}")
    print(f"  Train:             {len(train_df):,} ({TRAIN_RATIO*100:.1f}%)")
    print(f"  Validation:        {len(val_df):,} ({VAL_RATIO*100:.1f}%)")
    print(f"  Test:              {len(test_df):,} ({(1-TRAIN_RATIO-VAL_RATIO)*100:.1f}%)")
    print()
    print(f"Threat Distribution (SHOULD ALL BE ~{threat_pct:.2f}%):")
    print(f"  Train:     {train_threats:,}/{len(train_df):,} = {train_threat_pct:.2f}%")
    print(f"  Val:       {val_threats:,}/{len(val_df):,} = {val_threat_pct:.2f}%")
    print(f"  Test:      {test_threats:,}/{len(test_df):,} = {test_threat_pct:.2f}%")
    print()

    # Verify stratification worked
    print("✅ Verification:")
    max_diff = max(
        abs(train_threat_pct - val_threat_pct),
        abs(train_threat_pct - test_threat_pct),
        abs(val_threat_pct - test_threat_pct)
    )

    if max_diff < 0.5:
        print(f"  ✓ All splits have consistent threat distribution!")
        print(f"  ✓ Maximum difference: {max_diff:.3f}% (< 0.5% threshold)")
    else:
        print(f"  ⚠️  Warning: Splits have different distributions!")
        print(f"  ⚠️  Maximum difference: {max_diff:.3f}%")
    print()

    # Save splits
    print("💾 Saving splits...")
    splits_path = Path(SPLITS_DIR)
    splits_path.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(f'{SPLITS_DIR}/train.parquet', index=False)
    print(f"  ✓ Saved {SPLITS_DIR}/train.parquet")

    val_df.to_parquet(f'{SPLITS_DIR}/val.parquet', index=False)
    print(f"  ✓ Saved {SPLITS_DIR}/val.parquet")

    test_df.to_parquet(f'{SPLITS_DIR}/test.parquet', index=False)
    print(f"  ✓ Saved {SPLITS_DIR}/test.parquet")
    print()

    # Save metadata
    metadata = {
        'total_records': len(full_df),
        'train_records': len(train_df),
        'val_records': len(val_df),
        'test_records': len(test_df),
        'train_threats': int(train_threats),
        'val_threats': int(val_threats),
        'test_threats': int(test_threats),
        'train_threat_pct': float(train_threat_pct),
        'val_threat_pct': float(val_threat_pct),
        'test_threat_pct': float(test_threat_pct),
        'overall_threat_pct': float(threat_pct),
        'split_method': 'stratified',
        'random_state': RANDOM_STATE,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'max_distribution_diff': float(max_diff)
    }

    metadata_file = f"{METADATA_DIR}/split_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_file}")
    print()

    print("=" * 80)
    print("✅ STRATIFIED SPLITS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Update training config: entropy_coef=0.05, lr=3e-4, value_coef=1.5")
    print("2. Run training and monitor for positive rewards")
    print()

if __name__ == "__main__":
    main()
