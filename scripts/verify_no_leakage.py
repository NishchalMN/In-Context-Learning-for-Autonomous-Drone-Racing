"""
Verify no data leakage between train and validation sets.

Checks:
1. Original base episodes used in train vs val are disjoint
2. No overlapping episode IDs between train and val
"""

import h5py
import argparse


def extract_base_episode_id(ep_name):
    """
    Extract base episode ID from augmented episode name.

    Examples:
    - ep_000000 -> ep_000000
    - ep_000000_rot90 -> ep_000000
    - ep_000000_rev -> ep_000000
    - ep_000000_translated -> ep_000000
    """
    # Remove augmentation suffixes
    base = ep_name.split('_rot')[0]
    base = base.split('_rev')[0]
    base = base.split('_sub')[0]
    base = base.split('_trans')[0]
    base = base.split('_speed')[0]
    base = base.split('_noisy')[0]

    return base


def verify_no_leakage(train_path, val_path):
    """Verify no data leakage between train and val."""

    print("=" * 70)
    print("VERIFYING NO DATA LEAKAGE")
    print("=" * 70)
    print()

    # Load train episodes
    with h5py.File(train_path, 'r') as f:
        if 'episodes' in f.keys():
            train_episodes = sorted([k for k in f['episodes'].keys() if k.startswith('ep_')])
        else:
            train_episodes = sorted([k for k in f.keys() if k.startswith('ep_')])

    # Load val episodes
    with h5py.File(val_path, 'r') as f:
        if 'episodes' in f.keys():
            val_episodes = sorted([k for k in f['episodes'].keys() if k.startswith('ep_')])
        else:
            val_episodes = sorted([k for k in f.keys() if k.startswith('ep_')])

    print(f"Train episodes: {len(train_episodes)}")
    print(f"Val episodes: {len(val_episodes)}")
    print()

    # Extract base episode IDs
    train_base = set([extract_base_episode_id(ep) for ep in train_episodes])
    val_base = set([extract_base_episode_id(ep) for ep in val_episodes])

    print(f"Unique base episodes in train: {len(train_base)}")
    print(f"Unique base episodes in val: {len(val_base)}")
    print()

    # Check for overlap
    overlap = train_base & val_base

    if len(overlap) > 0:
        print("❌ DATA LEAKAGE DETECTED!")
        print(f"   {len(overlap)} base episodes appear in both train and val:")
        for ep in sorted(overlap):
            print(f"      - {ep}")
        print()
        print("   This means augmented variants of the same flight appear in")
        print("   both training and validation, causing data leakage!")
        return False
    else:
        print("✅ NO DATA LEAKAGE!")
        print("   Train and validation sets are completely disjoint.")
        print("   No base episodes are shared between train and val.")
        print()
        print(f"   Train base episodes: {sorted(train_base)[:5]}...")
        print(f"   Val base episodes:   {sorted(val_base)}")
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/train_final.h5')
    parser.add_argument('--val', default='data/val_final.h5')

    args = parser.parse_args()
    verify_no_leakage(args.train, args.val)
