"""
Split base episodes into train/val BEFORE augmentation.

This ensures no data leakage - validation episodes are completely unseen.
"""

import h5py
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_episodes(input_path, train_output, val_output, split_ratio=0.8, seed=42):
    """Split episodes into train/val sets."""

    print(f"Reading from: {input_path}")

    with h5py.File(input_path, 'r') as f_in:
        # Get episodes
        if 'episodes' in f_in.keys():
            ep_group = f_in['episodes']
            prefix = 'episodes/'
        else:
            ep_group = f_in
            prefix = ''

        episodes = sorted([k for k in ep_group.keys() if k.startswith('ep_')])
        print(f"\nTotal episodes: {len(episodes)}")

        # Split episodes
        train_eps, val_eps = train_test_split(
            episodes,
            train_size=split_ratio,
            random_state=seed,
            shuffle=True
        )

        print(f"Train episodes: {len(train_eps)}")
        print(f"Val episodes: {len(val_eps)}")

        # Create train file
        print(f"\nWriting training set to: {train_output}")
        with h5py.File(train_output, 'w') as f_train:
            train_group = f_train.create_group('episodes')
            for ep_name in tqdm(train_eps, desc="Train"):
                ep_src = ep_group[ep_name]
                f_in.copy(prefix + ep_name, train_group, name=ep_name)

        # Create val file
        print(f"\nWriting validation set to: {val_output}")
        with h5py.File(val_output, 'w') as f_val:
            val_group = f_val.create_group('episodes')
            for ep_name in tqdm(val_eps, desc="Val"):
                ep_src = ep_group[ep_name]
                f_in.copy(prefix + ep_name, val_group, name=ep_name)

    print("\n" + "="*70)
    print("Split complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Augment ONLY training set:")
    print(f"   python scripts/augment_dataset_fast.py --input {train_output} --output data/train_augmented.h5")
    print(f"\n2. Expand both sets with sliding windows:")
    print(f"   python scripts/expand_dataset_sliding_window.py --input data/train_augmented.h5 --output data/train_final.h5")
    print(f"   python scripts/expand_dataset_sliding_window.py --input {val_output} --output data/val_final.h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/ratm_racing_dataset.h5')
    parser.add_argument('--train-output', default='data/train_base.h5')
    parser.add_argument('--val-output', default='data/val_base.h5')
    parser.add_argument('--split-ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    split_episodes(args.input, args.train_output, args.val_output, args.split_ratio, args.seed)
