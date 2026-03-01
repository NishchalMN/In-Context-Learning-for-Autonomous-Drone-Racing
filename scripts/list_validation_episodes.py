"""
List validation episodes per track for easy reference.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_icl.dataset_icl import ICLRacingDataset
import numpy as np

def main():
    # Load dataset
    dataset = ICLRacingDataset(
        h5_path='data/ratm_racing_dataset.h5',
        num_demos=3,
        max_seq_len=512,
        use_images=False,
        normalize=True,
        subsample_factor=10
    )

    # Get validation split (same as training: 20%, seed 42)
    all_episodes = dataset.episodes

    # Group by base episode (remove window suffix from expanded dataset)
    base_episodes = {}
    for ep in all_episodes:
        # Check if it's from expanded dataset with _wXXX suffix
        if '_w' in ep and ep.split('_')[-1].startswith('w'):
            base_name = '_'.join(ep.split('_')[:-1])
        else:
            base_name = ep

        if base_name not in base_episodes:
            base_episodes[base_name] = []
        base_episodes[base_name].append(ep)

    # Get base episode names
    base_ep_list = sorted(list(base_episodes.keys()))

    # Simulate validation split (20%, seed=42) - same as training
    np.random.seed(42)
    split_idx = int(len(base_ep_list) * 0.8)
    np.random.shuffle(base_ep_list)

    train_base = base_ep_list[:split_idx]
    val_base = base_ep_list[split_idx:]

    # Group by track
    track_val_episodes = {}
    for base_ep in val_base:
        # Get track from first window/full episode
        first_ep = base_episodes[base_ep][0]
        track = dataset.episode_to_track.get(first_ep, 'unknown')

        if track not in track_val_episodes:
            track_val_episodes[track] = []
        track_val_episodes[track].append(base_ep)

    # Print results
    print("="*60)
    print("VALIDATION EPISODES BY TRACK")
    print("="*60)
    print(f"Using same split as training: 20% validation, seed=42")
    print(f"Total validation episodes: {len(val_base)}")
    print("="*60)

    for track in sorted(track_val_episodes.keys()):
        eps = sorted(track_val_episodes[track])
        print(f"\n{track.upper()} Track ({len(eps)} validation episodes):")
        print("-"*60)
        for ep in eps:
            print(f"  {ep}")

    print("\n" + "="*60)
    print("EXAMPLE COMMANDS FOR EACH TRACK")
    print("="*60)

    for track in sorted(track_val_episodes.keys()):
        eps = sorted(track_val_episodes[track])
        if len(eps) >= 4:
            target = eps[0]
            demos = eps[1:4]

            print(f"\n{track.upper()} Track:")
            print(f"  Target: {target}")
            print(f"  Demos:  {', '.join(demos)}")
            print(f"\n  Command:")
            print(f"  python3 scripts/create_demo_video.py \\")
            print(f"      --checkpoint checkpoints/ratm_icl_expanded/best_model.pt \\")
            print(f"      --data data/ratm_racing_dataset.h5 \\")
            print(f"      --target-episode {target} \\")
            print(f"      --demo-episodes {' '.join(demos)} \\")
            print(f"      --device cuda \\")
            print(f"      --demo-color red \\")
            print(f"      --frame-skip 4 \\")
            print(f"      --output demo_{track}_{target}.mp4")

if __name__ == '__main__':
    main()
