"""
check_real_dataset_files.py
===========================
Quick sanity check: verifies that the required Rossmann raw files exist
and prints basic info (size, columns, first 5 rows).
"""

import os
import sys
import pandas as pd


def sizeof_fmt(num_bytes):
    """Human-readable file size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def main():
    train_path = 'data/real/raw/train.csv'
    store_path = 'data/real/raw/store.csv'

    files = {'train.csv': train_path, 'store.csv': store_path}
    all_ok = True

    for name, path in files.items():
        if os.path.exists(path):
            print(f"[OK]    {name} found at {path}")
        else:
            print(f"[MISSING] {name} NOT found at {path}")
            all_ok = False

    if not all_ok:
        print("\nPlease download the Rossmann Store Sales dataset and place")
        print("train.csv and store.csv inside data/real/raw/")
        sys.exit(1)

    print("\nBoth files present.\n")

    for name, path in files.items():
        size = os.path.getsize(path)
        print(f"--- {name} ({sizeof_fmt(size)}) ---")
        df = pd.read_csv(path, low_memory=False, nrows=5)
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
