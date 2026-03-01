"""
Convert RATM dataset to HDF5 format compatible with ICL training.

This script converts the TII Racing "Race Against the Machine" dataset
from CSV format to the HDF5 format expected by our ICL transformer.
"""

import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
import argparse
from tqdm import tqdm


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (rad) to quaternion."""
    r = Rotation.from_euler('xyz', np.column_stack([roll, pitch, yaw]))
    return r.as_quat()  # Returns [x, y, z, w]


def normalize_control_inputs(channels_roll, channels_pitch, channels_thrust, channels_yaw):
    """
    Normalize RC channels from [1000, 2000] to action space.

    Returns: [thrust, roll, pitch, yaw] in normalized range
    """
    # Normalize from [1000, 2000] to [-1, 1] for roll/pitch/yaw
    # and [0, 1] for thrust
    roll_norm = (channels_roll - 1500) / 500.0
    pitch_norm = (channels_pitch - 1500) / 500.0
    yaw_norm = (channels_yaw - 1500) / 500.0
    thrust_norm = (channels_thrust - 1000) / 1000.0

    # Stack as [thrust, roll, pitch, yaw]
    return np.column_stack([thrust_norm, roll_norm, pitch_norm, yaw_norm])


def extract_gate_positions(df):
    """Extract unique gate positions from dataframe."""
    # Get gate center positions
    gates = []
    for gate_num in range(1, 5):
        x_col = f'gate{gate_num}_int_x'
        y_col = f'gate{gate_num}_int_y'
        z_col = f'gate{gate_num}_int_z'

        if x_col in df.columns:
            # Use median position across trajectory
            gate_pos = np.array([
                df[x_col].median(),
                df[y_col].median(),
                df[z_col].median()
            ])
            gates.append(gate_pos)

    return np.array(gates)  # (num_gates, 3)


def convert_csv_to_episode(csv_path, use_500hz=True):
    """
    Convert a single RATM CSV file to episode data.

    Args:
        csv_path: Path to CSV file
        use_500hz: If True, use 500Hz CSV; else use camera timestamp sync CSV

    Returns:
        Dictionary with episode data
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Extract state components
    pos = df[['drone_x', 'drone_y', 'drone_z']].values  # (T, 3)

    # Convert Euler angles to quaternions
    roll = df['drone_roll'].values
    pitch = df['drone_pitch'].values
    yaw = df['drone_yaw'].values
    quat = euler_to_quaternion(roll, pitch, yaw)  # (T, 4)

    # Velocities (already in CSV)
    lin_vel = df[['drone_velocity_linear_x', 'drone_velocity_linear_y', 'drone_velocity_linear_z']].values
    ang_vel = df[['drone_velocity_angular_x', 'drone_velocity_angular_y', 'drone_velocity_angular_z']].values

    # IMU data
    imu_acc = df[['accel_x', 'accel_y', 'accel_z']].values
    imu_gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values

    # Control inputs (RC channels)
    actions = normalize_control_inputs(
        df['channels_roll'].values,
        df['channels_pitch'].values,
        df['channels_thrust'].values,
        df['channels_yaw'].values
    )

    # Extract gate positions
    gates = extract_gate_positions(df)

    # Create gates_passed array (dummy for now - would need gate detection logic)
    # For now, just mark as 0 (not passed yet)
    gates_passed = np.zeros(len(df), dtype=np.int32)

    # Timestamps
    timestamps = df['elapsed_time'].values

    return {
        'state/pos': pos.astype(np.float32),
        'state/quat': quat.astype(np.float32),
        'state/lin_vel': lin_vel.astype(np.float32),
        'state/ang_vel': ang_vel.astype(np.float32),
        'imu/lin_acc': imu_acc.astype(np.float32),
        'imu/ang_vel': imu_gyro.astype(np.float32),
        'actions': actions.astype(np.float32),
        'track_gates': gates.astype(np.float32),
        'gates_passed': gates_passed,
        'timestamps': timestamps.astype(np.float32)
    }


def convert_ratm_dataset(
    ratm_root: str,
    output_path: str,
    use_500hz: bool = True,
    include_piloted: bool = True,
    include_autonomous: bool = True
):
    """
    Convert entire RATM dataset to HDF5.

    Args:
        ratm_root: Root directory of RATM dataset
        output_path: Output HDF5 file path
        use_500hz: Use 500Hz CSV files (True) or camera sync CSV (False)
        include_piloted: Include human-piloted flights
        include_autonomous: Include autonomous flights
    """
    ratm_root = Path(ratm_root)

    # Collect all flight directories
    flights = []

    if include_piloted:
        piloted_dir = ratm_root / 'piloted'
        if piloted_dir.exists():
            flights.extend([(piloted_dir / f, 'piloted') for f in os.listdir(piloted_dir)
                           if f.startswith('flight-') and (piloted_dir / f).is_dir()])

    if include_autonomous:
        autonomous_dir = ratm_root / 'autonomous'
        if autonomous_dir.exists():
            flights.extend([(autonomous_dir / f, 'autonomous') for f in os.listdir(autonomous_dir)
                           if f.startswith('flight-') and (autonomous_dir / f).is_dir()])

    print(f"Found {len(flights)} flights to convert")

    # Create HDF5 file
    with h5py.File(output_path, 'w') as h5f:
        episodes_group = h5f.create_group('episodes')

        episode_idx = 0

        for flight_dir, flight_type in tqdm(flights, desc="Converting flights"):
            flight_name = flight_dir.name

            # Find CSV file
            if use_500hz:
                csv_file = flight_dir / f"{flight_name}_500hz_freq_sync.csv"
            else:
                csv_file = flight_dir / f"{flight_name}_cam_ts_sync.csv"

            if not csv_file.exists():
                print(f"Warning: CSV not found for {flight_name}, skipping")
                continue

            try:
                # Convert CSV to episode data
                episode_data = convert_csv_to_episode(str(csv_file), use_500hz=use_500hz)

                # Create episode group
                ep_name = f"ep_{episode_idx:06d}"
                ep_group = episodes_group.create_group(ep_name)

                # Store metadata as attributes
                ep_group.attrs['flight_name'] = flight_name
                ep_group.attrs['flight_type'] = flight_type

                # Extract track type from flight name
                if 'ellipse' in flight_name:
                    track_type = 'ellipse'
                elif 'lemniscate' in flight_name:
                    track_type = 'lemniscate'
                elif 'trackRATM' in flight_name:
                    track_type = 'trackRATM'
                else:
                    track_type = 'unknown'
                ep_group.attrs['track_type'] = track_type

                # Write datasets
                for key, data in episode_data.items():
                    if '/' in key:
                        # Nested group (e.g., 'state/pos')
                        parent, child = key.split('/')
                        if parent not in ep_group:
                            parent_group = ep_group.create_group(parent)
                        else:
                            parent_group = ep_group[parent]
                        parent_group.create_dataset(child, data=data, compression='gzip')
                    else:
                        ep_group.create_dataset(key, data=data, compression='gzip')

                episode_idx += 1

            except Exception as e:
                print(f"Error converting {flight_name}: {e}")
                continue

        print(f"\nSuccessfully converted {episode_idx} episodes")
        print(f"Output: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Convert RATM dataset to HDF5')
    parser.add_argument('--ratm-root', type=str, required=True,
                       help='Root directory of RATM dataset (contains piloted/ and autonomous/)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output HDF5 file path')
    parser.add_argument('--use-500hz', action='store_true', default=True,
                       help='Use 500Hz CSV files (default: True)')
    parser.add_argument('--camera-sync', action='store_true',
                       help='Use camera timestamp sync CSV instead of 500Hz')
    parser.add_argument('--piloted-only', action='store_true',
                       help='Convert only piloted flights')
    parser.add_argument('--autonomous-only', action='store_true',
                       help='Convert only autonomous flights')

    args = parser.parse_args()

    use_500hz = not args.camera_sync
    include_piloted = not args.autonomous_only
    include_autonomous = not args.piloted_only

    convert_ratm_dataset(
        ratm_root=args.ratm_root,
        output_path=args.output,
        use_500hz=use_500hz,
        include_piloted=include_piloted,
        include_autonomous=include_autonomous
    )


if __name__ == '__main__':
    main()
