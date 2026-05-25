#!/usr/bin/env python3
"""
Create train/val/test split parquet files for BF-C2DL-HSC and BF-C2DL-MuSC.

Generates 4 parquet files:
  data/dataframes/BF-C2DL-HSC/whole_image/BF-C2DL-HSC_split_fold-1.parquet
  data/dataframes/BF-C2DL-HSC/whole_image/BF-C2DL-HSC_split_fold-2.parquet
  data/dataframes/BF-C2DL-MuSC/whole_image/BF-C2DL-MuSC_split_fold-1.parquet
  data/dataframes/BF-C2DL-MuSC/whole_image/BF-C2DL-MuSC_split_fold-2.parquet

Split strategy (default):
  - Fold 1: Train/Val on sequence 01, Test on sequence 02
  - Fold 2: Train/Val on sequence 02, Test on sequence 01
  - Within train/val sequence: 80% train, 20% val (random, seed=42)

NOTE: This is a standard cross-validation split. Please verify it matches
your paper's methodology before training. If you need a different split
(e.g., specific frames for test, or object-level splits), adjust the
create_fold_split() function accordingly.
"""

import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def extract_frame_number(filename):
    """Extract numeric frame ID from filenames like t0000.tif or man_seg0058.tif."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1


def scan_sequence(dataset_dir, seq_id):
    """
    Scan a sequence folder (e.g., '01') and its GT folder (e.g., '01_GT/SEG')
    to collect image paths and GT mask paths.
    """
    seq_dir = dataset_dir / seq_id
    gt_dir = dataset_dir / f"{seq_id}_GT" / "SEG"
    
    if not seq_dir.exists():
        return []
    
    # Build lookup of GT masks by frame number
    gt_masks = {}
    if gt_dir.exists():
        for gt_path in sorted(gt_dir.glob("*.tif")):
            frame_num = extract_frame_number(gt_path.name)
            gt_masks[frame_num] = gt_path
    
    rows = []
    for img_path in sorted(seq_dir.glob("*.tif")):
        frame_num = extract_frame_number(img_path.name)
        gt_path = gt_masks.get(frame_num)
        
        rows.append({
            "frame_id": frame_num,
            "sequence": seq_id,
            "image_path": str(img_path),
            "gt_mask_path": str(gt_path) if gt_path else None,
            "has_gt": gt_path is not None,
        })
    
    return rows


def create_fold_split(dataset_name, dataset_dir, output_dir, 
                      train_val_seq, test_seq, fold_num, random_seed=42):
    """
    Create a single fold split parquet file.
    """
    # Scan sequences
    train_val_rows = scan_sequence(dataset_dir, train_val_seq)
    test_rows = scan_sequence(dataset_dir, test_seq)
    
    if not train_val_rows:
        raise ValueError(f"No images found in {dataset_dir}/{train_val_seq}")
    if not test_rows:
        raise ValueError(f"No images found in {dataset_dir}/{test_seq}")
    
    # Split train/val
    df_train_val = pd.DataFrame(train_val_rows)
    n_train = max(1, int(0.8 * len(df_train_val)))
    
    rng = np.random.RandomState(random_seed + fold_num)
    shuffled_idx = rng.permutation(len(df_train_val))
    
    train_idx = shuffled_idx[:n_train]
    val_idx = shuffled_idx[n_train:]
    
    df_train_val.loc[train_idx, "split"] = "train"
    df_train_val.loc[val_idx, "split"] = "val"
    
    # Test set
    df_test = pd.DataFrame(test_rows)
    df_test["split"] = "test"
    
    # Combine
    df = pd.concat([df_train_val, df_test], ignore_index=True)
    
    # Ensure correct column order
    df = df[["frame_id", "sequence", "image_path", "gt_mask_path", "has_gt", "split"]]
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_split_fold-{fold_num}.parquet"
    
    if output_path.exists():
        print(f"Warning: {output_path} already exists. Overwriting.")
    
    df.to_parquet(output_path, index=False)
    
    print(f"\nCreated: {output_path}")
    print(f"  Total: {len(df)} frames")
    print(f"  Train: {sum(df['split'] == 'train')}, Val: {sum(df['split'] == 'val')}, Test: {sum(df['split'] == 'test')}")
    print(f"  With GT: {sum(df['has_gt'])} / {len(df)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    
    return df


def main():
    repo_root = Path(__file__).parent
    if not (repo_root / "data" / "synchronized_data").exists():
        print("ERROR: Please run this script from the repo root directory.")
        sys.exit(1)
    
    base_data_dir = repo_root / "data" / "synchronized_data"
    output_base = repo_root / "data" / "dataframes"
    
    datasets = [
        ("BF-C2DL-HSC", base_data_dir / "BF-C2DL-HSC"),
        ("BF-C2DL-MuSC", base_data_dir / "BF-C2DL-MuSC"),
    ]
    
    for dataset_name, dataset_dir in datasets:
        if not dataset_dir.exists():
            print(f"Warning: {dataset_dir} not found, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}")
        print(f"{'='*60}")
        
        # Fold 1: Train/Val on 01, Test on 02
        create_fold_split(
            dataset_name, dataset_dir, 
            output_base / dataset_name / "whole_image",
            train_val_seq="01", test_seq="02", fold_num=1
        )
        
        # Fold 2: Train/Val on 02, Test on 01
        create_fold_split(
            dataset_name, dataset_dir,
            output_base / dataset_name / "whole_image",
            train_val_seq="02", test_seq="01", fold_num=2
        )


if __name__ == "__main__":
    main()
