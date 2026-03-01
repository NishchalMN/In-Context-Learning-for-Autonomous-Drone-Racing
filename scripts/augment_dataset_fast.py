"""
Fast Data Augmentation for ICL Drone Racing

Augments existing dataset with:
1. Track reversal (2x data)
2. Temporal subsampling (3x data)
3. Small perturbations
4. Track rotation (4x data) - NEW!
5. Track translation (2x data) - NEW!
6. Speed variation (3x data) - NEW!

Total: Up to 50x data in < 1 hour

Usage:
    python scripts/augment_dataset_fast.py \
        --input data/ratm_racing_dataset.h5 \
        --output data/ratm_racing_dataset_augmented.h5
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d


def reverse_trajectory(states, actions):
    """Reverse trajectory - fly track backwards."""
    # Flip time dimension
    states_rev = np.flip(states, axis=0).copy()
    actions_rev = np.flip(actions, axis=0).copy()

    # Reverse velocities and actions
    states_rev[:, 7:10] *= -1  # Linear velocity
    states_rev[:, 10:13] *= -1  # Angular velocity
    actions_rev *= -1  # Actions

    return states_rev, actions_rev


def temporal_subsample(states, actions, rate=2):
    """Subsample at lower frequency."""
    return states[::rate], actions[::rate]


def add_small_noise(states, actions, noise_scale=0.01):
    """Add small Gaussian noise to states."""
    noise_state = np.random.randn(*states.shape) * noise_scale
    noise_action = np.random.randn(*actions.shape) * (noise_scale * 0.5)

    return states + noise_state, actions + noise_action


def rotate_trajectory(states, actions, angle, gates=None):
    """
    Rotate entire trajectory around Z-axis.

    Args:
        states: (T, 19) - [pos(3), quat(4), lin_vel(3), ang_vel(3), imu_acc(3), imu_gyro(3)]
        actions: (T, 4) - [thrust, omega_x, omega_y, omega_z]
        angle: rotation angle in radians
        gates: (N, 3) - gate positions (optional)

    Returns:
        states_rot, actions, gates_rot
    """
    states_rot = states.copy()

    # Rotation matrix around Z-axis
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])

    # Rotate positions (indices 0:3)
    states_rot[:, 0:3] = states[:, 0:3] @ R.T

    # Rotate linear velocity (indices 7:10)
    states_rot[:, 7:10] = states[:, 7:10] @ R.T

    # Rotate IMU linear acceleration (indices 13:16)
    states_rot[:, 13:16] = states[:, 13:16] @ R.T

    # Rotate quaternion (indices 3:7)
    # For rotation around Z by angle, multiply by quat [cos(a/2), 0, 0, sin(a/2)]
    qz = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    for t in range(len(states)):
        q = states[t, 3:7]
        states_rot[t, 3:7] = quaternion_multiply(qz, q)

    # Rotate gates (CRITICAL: must rotate gates to match trajectory!)
    gates_rot = None
    if gates is not None:
        gates_rot = gates @ R.T

    # Actions: thrust unchanged, rotate body rates
    # Body rates in body frame, so they rotate with the body - actually no change needed
    # The drone's reference frame rotates with it

    return states_rot, actions, gates_rot


def quaternion_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def translate_trajectory(states, translation, gates=None):
    """
    Translate entire trajectory in 3D space.

    Args:
        states: (T, 19)
        translation: (3,) offset in x, y, z
        gates: (N, 3) - gate positions (optional)

    Returns:
        states_trans, gates_trans
    """
    states_trans = states.copy()
    states_trans[:, 0:3] += translation

    # Translate gates (CRITICAL: must translate gates to match trajectory!)
    gates_trans = None
    if gates is not None:
        gates_trans = gates + translation

    return states_trans, gates_trans


def speed_variation(states, actions, speed_factor):
    """
    Speed up or slow down trajectory.

    Args:
        states: (T, 19)
        actions: (T, 4)
        speed_factor: > 1 speeds up, < 1 slows down
    """
    T = len(states)
    new_T = max(int(T / speed_factor), 50)  # Minimum 50 frames

    # Original and new time indices
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, new_T)

    # Interpolate each dimension of states
    new_states = np.zeros((new_T, states.shape[1]))
    for i in range(states.shape[1]):
        f = interp1d(old_t, states[:, i], kind='linear', fill_value='extrapolate')
        new_states[:, i] = f(new_t)

    # Scale velocities by speed factor
    new_states[:, 7:10] *= speed_factor   # Linear velocity
    new_states[:, 10:13] *= speed_factor  # Angular velocity

    # Interpolate actions
    new_actions = np.zeros((new_T, actions.shape[1]))
    for i in range(actions.shape[1]):
        f = interp1d(old_t, actions[:, i], kind='linear', fill_value='extrapolate')
        new_actions[:, i] = f(new_t)

    return new_states, new_actions


def augment_dataset(input_path, output_path, enable_rotation=True, enable_translation=True,
                     enable_speed=True, rotation_angles=None, speed_factors=None):
    """
    Augment HDF5 dataset with multiple augmentation strategies.

    Args:
        input_path: Input HDF5 file
        output_path: Output HDF5 file
        enable_rotation: Enable track rotation augmentation
        enable_translation: Enable track translation augmentation
        enable_speed: Enable speed variation augmentation
        rotation_angles: List of rotation angles in degrees (default: [90, 180, 270])
        speed_factors: List of speed factors (default: [0.8, 1.2])
    """
    if rotation_angles is None:
        rotation_angles = [90, 180, 270]  # Degrees
    if speed_factors is None:
        speed_factors = [0.8, 1.2]  # Slow and fast

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"\nAugmentation settings:")
    print(f"  Rotation: {enable_rotation} (angles: {rotation_angles})")
    print(f"  Translation: {enable_translation}")
    print(f"  Speed variation: {enable_speed} (factors: {speed_factors})")

    # Open input
    with h5py.File(input_path, 'r') as f_in:
        # Get original episodes - check if episodes are in a group or at root
        if 'episodes' in f_in.keys():
            episodes = sorted([k for k in f_in['episodes'].keys() if k.startswith('ep_')])
            episodes_group = 'episodes/'
        else:
            episodes = sorted([k for k in f_in.keys() if k.startswith('ep_')])
            episodes_group = ''

        # Check if already expanded (has window suffix)
        is_expanded = any('_' in ep.split('_', 2)[-1] for ep in episodes[:10])

        print(f"\nFound {len(episodes)} episodes")
        print(f"Dataset type: {'Expanded (with windows)' if is_expanded else 'Original'}")

        # Create output
        with h5py.File(output_path, 'w') as f_out:
            augmentation_count = {
                'original': 0,
                'reversed': 0,
                'subsampled_2x': 0,
                'rotated': 0,
                'translated': 0,
                'speed_varied': 0,
                'noisy': 0
            }

            for ep_name in tqdm(episodes, desc="Augmenting episodes"):
                ep = f_in[episodes_group + ep_name]
                track = ep.attrs.get('track', 'unknown')

                # Extract data
                states = np.concatenate([
                    np.array(ep['state']['pos']),
                    np.array(ep['state']['quat']),
                    np.array(ep['state']['lin_vel']),
                    np.array(ep['state']['ang_vel']),
                    np.array(ep['imu']['lin_acc']),
                    np.array(ep['imu']['ang_vel'])
                ], axis=-1)

                actions = np.array(ep['actions'])

                # Extract gate positions (CRITICAL for gate-aware training)
                gates = None
                if 'track_gates' in ep:
                    gates = np.array(ep['track_gates'])

                # Ensure 2D arrays
                if states.ndim == 1:
                    continue  # Skip malformed episodes

                # 1. Original episode
                create_episode(f_out, ep_name, states, actions, track, gates)
                augmentation_count['original'] += 1

                # 2. Reversed trajectory (gates unchanged - same track, different direction)
                states_rev, actions_rev = reverse_trajectory(states, actions)
                create_episode(f_out, f"{ep_name}_rev", states_rev, actions_rev, track, gates)
                augmentation_count['reversed'] += 1

                # 3. Temporal subsampling (2x) - gates unchanged
                states_sub2, actions_sub2 = temporal_subsample(states, actions, rate=2)
                if len(states_sub2) > 100:
                    create_episode(f_out, f"{ep_name}_sub2", states_sub2, actions_sub2, track, gates)
                    augmentation_count['subsampled_2x'] += 1

                # 4. Track rotations (90°, 180°, 270°) - MUST rotate gates too!
                if enable_rotation:
                    for angle_deg in rotation_angles:
                        angle_rad = np.deg2rad(angle_deg)
                        states_rot, actions_rot, gates_rot = rotate_trajectory(states, actions, angle_rad, gates)
                        create_episode(f_out, f"{ep_name}_rot{angle_deg}", states_rot, actions_rot, track, gates_rot)
                        augmentation_count['rotated'] += 1

                # 5. Track translation (random offset) - MUST translate gates too!
                if enable_translation:
                    translation = np.random.uniform(-5, 5, size=3)
                    translation[2] = np.random.uniform(-1, 1)  # Smaller Z offset
                    states_trans, gates_trans = translate_trajectory(states, translation, gates)
                    create_episode(f_out, f"{ep_name}_trans", states_trans, actions, track, gates_trans)
                    augmentation_count['translated'] += 1

                # 6. Speed variation - gates unchanged (same positions, different speed)
                if enable_speed:
                    for speed_factor in speed_factors:
                        states_speed, actions_speed = speed_variation(states, actions, speed_factor)
                        speed_label = f"fast{int(speed_factor*100)}" if speed_factor > 1 else f"slow{int(speed_factor*100)}"
                        create_episode(f_out, f"{ep_name}_{speed_label}", states_speed, actions_speed, track, gates)
                        augmentation_count['speed_varied'] += 1

                # 7. Small noise (30% of episodes) - gates unchanged
                if np.random.random() < 0.3:
                    states_noisy, actions_noisy = add_small_noise(states, actions)
                    create_episode(f_out, f"{ep_name}_noise", states_noisy, actions_noisy, track, gates)
                    augmentation_count['noisy'] += 1

    print(f"\n{'='*70}")
    print("Augmentation Summary:")
    print(f"{'='*70}")
    for aug_type, count in augmentation_count.items():
        print(f"  {aug_type}: {count} episodes")
    total = sum(augmentation_count.values())
    print(f"\nTotal episodes: {total}")
    print(f"Augmentation factor: {total / max(augmentation_count['original'], 1):.1f}x")


def create_episode(f, ep_name, states, actions, track, gates=None):
    """Create episode in HDF5 file with gate information."""
    grp = f.create_group(ep_name)
    grp.attrs['track'] = track

    # Split states into components (assuming 19D state)
    grp.create_dataset('state/pos', data=states[:, 0:3])
    grp.create_dataset('state/quat', data=states[:, 3:7])
    grp.create_dataset('state/lin_vel', data=states[:, 7:10])
    grp.create_dataset('state/ang_vel', data=states[:, 10:13])
    grp.create_dataset('imu/lin_acc', data=states[:, 13:16])
    grp.create_dataset('imu/ang_vel', data=states[:, 16:19])

    # Split actions into components (4D action)
    grp.create_dataset('action/thrust', data=actions[:, 0])
    grp.create_dataset('action/omega', data=actions[:, 1:4])

    # Store gate positions (CRITICAL for gate-aware training)
    if gates is not None:
        grp.create_dataset('track_gates', data=gates)


def main():
    parser = argparse.ArgumentParser(description="Fast dataset augmentation")
    parser.add_argument('--input', type=str, default='data/ratm_racing_dataset.h5',
                        help='Input HDF5 file')
    parser.add_argument('--output', type=str, default='data/ratm_racing_dataset_augmented.h5',
                        help='Output HDF5 file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Augmentation toggles
    parser.add_argument('--no-rotation', action='store_true', help='Disable rotation augmentation')
    parser.add_argument('--no-translation', action='store_true', help='Disable translation augmentation')
    parser.add_argument('--no-speed', action='store_true', help='Disable speed variation augmentation')

    # Augmentation parameters
    parser.add_argument('--rotation-angles', type=int, nargs='+', default=[90, 180, 270],
                        help='Rotation angles in degrees (default: 90 180 270)')
    parser.add_argument('--speed-factors', type=float, nargs='+', default=[0.8, 1.2],
                        help='Speed variation factors (default: 0.8 1.2)')

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("="*70)
    print("FAST DATA AUGMENTATION")
    print("="*70)

    augment_dataset(
        args.input,
        args.output,
        enable_rotation=not args.no_rotation,
        enable_translation=not args.no_translation,
        enable_speed=not args.no_speed,
        rotation_angles=args.rotation_angles,
        speed_factors=args.speed_factors
    )

    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")
    print(f"\nAugmented dataset saved to: {args.output}")
    print("\nNext steps:")
    print("1. Expand with sliding windows:")
    print(f"   python scripts/expand_dataset_windows.py --input {args.output}")
    print("\n2. Then train on expanded+augmented dataset:")
    print(f"   python drone_icl/train.py --data <expanded_output> --epochs 150 ...")


if __name__ == '__main__':
    main()
