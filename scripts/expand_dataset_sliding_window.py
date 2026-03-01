"""
Expand RATM dataset using sliding window approach.

Takes 36 episodes and creates ~650 episodes by:
1. Splitting long trajectories into overlapping windows
2. Creating multiple starting points per episode
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def sliding_window_split(state_dict, actions, gates, imu_dict, timestamps,
                         window_size=200, stride=50):
    """
    Split a trajectory into overlapping windows.

    Args:
        state_dict: Dictionary with 'pos', 'quat', 'lin_vel', 'ang_vel'
        actions: (T, 4) action trajectory
        gates: (N, 3) gate positions
        imu_dict: Dictionary with 'lin_acc', 'ang_vel'
        timestamps: (T,) timestamps
        window_size: Length of each window
        stride: Step size between windows

    Returns:
        List of window dictionaries
    """
    T = len(actions)
    windows = []

    # Create overlapping windows
    for start_idx in range(0, T - window_size + 1, stride):
        end_idx = start_idx + window_size

        # Slice all state components
        state_window = {
            'pos': state_dict['pos'][start_idx:end_idx],
            'quat': state_dict['quat'][start_idx:end_idx],
            'lin_vel': state_dict['lin_vel'][start_idx:end_idx],
            'ang_vel': state_dict['ang_vel'][start_idx:end_idx]
        }

        # Slice IMU
        imu_window = {
            'lin_acc': imu_dict['lin_acc'][start_idx:end_idx],
            'ang_vel': imu_dict['ang_vel'][start_idx:end_idx]
        }

        # Keep all gates (they're global references)
        windows.append({
            'state': state_window,
            'actions': actions[start_idx:end_idx],
            'gates': gates,
            'imu': imu_window,
            'timestamps': timestamps[start_idx:end_idx],
            'start_idx': start_idx,
            'end_idx': end_idx
        })

    return windows


def expand_dataset(input_path, output_path, window_size=200, stride=50):
    """
    Create expanded dataset with sliding windows.

    Args:
        input_path: Original HDF5 dataset
        output_path: Output HDF5 path
        window_size: Window length
        stride: Stride between windows
    """
    print("="*60)
    print("Dataset Expansion - Sliding Window")
    print("="*60)

    # Load original dataset
    print(f"\nLoading original dataset: {input_path}")
    with h5py.File(input_path, 'r') as f_in:
        # Check if episodes are in a group or at root
        if 'episodes' in f_in.keys():
            episodes_group = f_in['episodes']
            episodes_prefix = 'episodes/'
        else:
            episodes_group = f_in
            episodes_prefix = ''

        episode_names = [k for k in episodes_group.keys() if k.startswith('ep_')]
        num_original = len(episode_names)
        print(f"Original episodes: {num_original}")

        # Track statistics
        track_stats = {}
        total_windows = 0

        # Create output dataset
        print(f"\nCreating expanded dataset: {output_path}")
        with h5py.File(output_path, 'w') as f_out:
            # Create episodes group to match input format
            episodes_out = f_out.create_group('episodes')

            # Process each original episode
            for ep_name in tqdm(episode_names, desc="Expanding episodes"):
                ep_group = episodes_group[ep_name]

                # Load data - handle both formats
                # Detect format: original has 'actions' dataset, augmented has 'action' subgroup
                is_original_format = 'actions' in ep_group

                if is_original_format:
                    # Original format with slashes (from ratm_racing_dataset.h5)
                    state_dict = {
                        'pos': ep_group['state/pos'][:],
                        'quat': ep_group['state/quat'][:],
                        'lin_vel': ep_group['state/lin_vel'][:],
                        'ang_vel': ep_group['state/ang_vel'][:]
                    }
                    imu_dict = {
                        'lin_acc': ep_group['imu/lin_acc'][:],
                        'ang_vel': ep_group['imu/ang_vel'][:]
                    }
                    actions = ep_group['actions'][:]
                    gates = ep_group['track_gates'][:]
                    timestamps = ep_group['timestamps'][:]
                    track_type = ep_group.attrs['track_type']
                else:
                    # Augmented format with subgroups (from augment_dataset_fast.py)
                    state_dict = {
                        'pos': ep_group['state']['pos'][:],
                        'quat': ep_group['state']['quat'][:],
                        'lin_vel': ep_group['state']['lin_vel'][:],
                        'ang_vel': ep_group['state']['ang_vel'][:]
                    }
                    imu_dict = {
                        'lin_acc': ep_group['imu']['lin_acc'][:],
                        'ang_vel': ep_group['imu']['ang_vel'][:]
                    }
                    # Reconstruct full actions from thrust + omega
                    thrust = ep_group['action']['thrust'][:]
                    omega = ep_group['action']['omega'][:]
                    if thrust.ndim == 1:
                        thrust = thrust[:, None]
                    actions = np.concatenate([thrust, omega], axis=-1)

                    # Read gates (CRITICAL for gate-aware training!)
                    if 'track_gates' in ep_group:
                        gates = ep_group['track_gates'][:]
                    else:
                        # Fallback for old augmented data without gates
                        print(f"Warning: No gates in {ep_name}, using dummy gates")
                        gates = np.array([[0, 0, 2], [10, 0, 2], [10, 10, 2], [0, 10, 2]])  # Dummy square

                    timestamps = np.arange(len(actions)) / 50.0  # 50Hz
                    track_type = ep_group.attrs.get('track', 'unknown')

                T = len(actions)

                # Track statistics
                if track_type not in track_stats:
                    track_stats[track_type] = {'original': 0, 'windows': 0}
                track_stats[track_type]['original'] += 1

                # Create sliding windows
                windows = sliding_window_split(
                    state_dict, actions, gates, imu_dict, timestamps,
                    window_size=window_size,
                    stride=stride
                )

                track_stats[track_type]['windows'] += len(windows)
                total_windows += len(windows)

                # Save each window as a new episode
                for i, window in enumerate(windows):
                    # Create unique episode name
                    new_ep_name = f"{ep_name}_w{i:03d}"

                    # Create group under episodes
                    new_group = episodes_out.create_group(new_ep_name)

                    # Create state subgroup
                    state_group = new_group.create_group('state')
                    state_group.create_dataset('pos', data=window['state']['pos'])
                    state_group.create_dataset('quat', data=window['state']['quat'])
                    state_group.create_dataset('lin_vel', data=window['state']['lin_vel'])
                    state_group.create_dataset('ang_vel', data=window['state']['ang_vel'])

                    # Create IMU subgroup
                    imu_group = new_group.create_group('imu')
                    imu_group.create_dataset('lin_acc', data=window['imu']['lin_acc'])
                    imu_group.create_dataset('ang_vel', data=window['imu']['ang_vel'])

                    # Save other data
                    new_group.create_dataset('actions', data=window['actions'])
                    new_group.create_dataset('track_gates', data=window['gates'])
                    new_group.create_dataset('timestamps', data=window['timestamps'])

                    # Save metadata
                    new_group.attrs['track_type'] = track_type
                    new_group.attrs['original_episode'] = ep_name
                    new_group.attrs['window_index'] = i
                    new_group.attrs['start_idx'] = window['start_idx']
                    new_group.attrs['end_idx'] = window['end_idx']
                    new_group.attrs['original_length'] = T

    # Print statistics
    print("\n" + "="*60)
    print("Expansion Complete")
    print("="*60)
    print(f"\nOriginal episodes: {num_original}")
    print(f"Expanded episodes: {total_windows}")
    print(f"Expansion factor: {total_windows / num_original:.1f}x\n")

    print("Per-track breakdown:")
    for track_type, stats in track_stats.items():
        print(f"  {track_type}:")
        print(f"    Original: {stats['original']}")
        print(f"    Windows:  {stats['windows']}")
        print(f"    Factor:   {stats['windows'] / stats['original']:.1f}x")

    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Expand dataset with sliding windows')
    parser.add_argument('--input', type=str,
                       default='data/ratm_racing_dataset.h5',
                       help='Input HDF5 dataset')
    parser.add_argument('--output', type=str,
                       default='data/ratm_racing_dataset_expanded.h5',
                       help='Output HDF5 dataset')
    parser.add_argument('--window-size', type=int, default=200,
                       help='Window size (timesteps)')
    parser.add_argument('--stride', type=int, default=50,
                       help='Stride between windows (timesteps)')

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)

    # Expand dataset
    expand_dataset(
        args.input,
        args.output,
        window_size=args.window_size,
        stride=args.stride
    )


if __name__ == '__main__':
    main()
