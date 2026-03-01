"""
Create clean demo videos for ICL drone racing.

Uses:
1. Original dataset (full laps, not windows)
2. Validation episodes only (honest generalization)
3. Ground truth states (perfect trajectory, no dynamics drift)
4. Complete laps with gate passing metrics

This produces publication-quality demo videos showing true generalization.
"""

import torch
import numpy as np
import h5py
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_icl.models import ICLTransformerPolicy
from drone_icl.dataset_icl import ICLRacingDataset


def get_validation_split(dataset, val_split=0.2, seed=42):
    """
    Get validation episodes using same split as training.

    Returns:
        train_episodes: List of training episode names
        val_episodes: List of validation episode names
    """
    # Group windows by original episode
    original_episodes = {}
    for ep_name in dataset.episodes:
        # For original dataset, episode names don't have _wXXX suffix
        # But we still group them in case
        if '_w' in ep_name:
            base_name = '_'.join(ep_name.split('_')[:-1])
        else:
            base_name = ep_name

        if base_name not in original_episodes:
            original_episodes[base_name] = []
        original_episodes[base_name].append(ep_name)

    # Split original episodes
    base_eps = sorted(list(original_episodes.keys()))
    np.random.seed(seed)
    np.random.shuffle(base_eps)

    split_idx = int(len(base_eps) * (1 - val_split))
    train_base = base_eps[:split_idx]
    val_base = base_eps[split_idx:]

    # Get all episode names
    train_episodes = []
    for base in train_base:
        train_episodes.extend(original_episodes[base])

    val_episodes = []
    for base in val_base:
        val_episodes.extend(original_episodes[base])

    return train_episodes, val_episodes


def select_demo_episodes(dataset, target_ep_name, train_episodes, val_episodes, num_demos=3):
    """
    Select demo episodes from validation set only.

    Args:
        dataset: ICLRacingDataset
        target_ep_name: Target episode to predict
        train_episodes: List of training episodes
        val_episodes: List of validation episodes
        num_demos: Number of demos to select

    Returns:
        demo_ep_names: List of demo episode names
    """
    target_track = dataset.episode_to_track[target_ep_name]

    # Get validation episodes from same track (excluding target)
    val_same_track = [ep for ep in val_episodes
                     if dataset.episode_to_track.get(ep) == target_track
                     and ep != target_ep_name]

    if len(val_same_track) < num_demos:
        print(f"⚠️  Warning: Only {len(val_same_track)} validation episodes for {target_track}")
        print(f"   Using all available validation episodes")
        return val_same_track

    # Randomly select demos
    demo_ep_names = np.random.choice(val_same_track, size=num_demos, replace=False).tolist()

    return demo_ep_names


def predict_actions_with_gt_states(model, demo_states, demo_actions, demo_masks,
                                   target_states, device='cpu', max_seq_len=512):
    """
    Predict actions using ground truth states (no dynamics).

    If target sequence is longer than max_seq_len, processes in chunks.

    Args:
        model: ICL policy model
        demo_states: (N, T_demo, 19) demonstration states
        demo_actions: (N, T_demo, 4) demonstration actions
        demo_masks: (N, T_demo) demonstration masks
        target_states: (T, 19) target ground truth states
        device: Device
        max_seq_len: Maximum sequence length model can handle

    Returns:
        pred_actions: (T, 4) predicted actions
    """
    model.eval()

    T = len(target_states)

    # If sequence fits in one pass
    if T <= max_seq_len:
        # Prepare input
        demo_states_in = demo_states.unsqueeze(0).to(device)  # (1, N, T_demo, 19)
        demo_actions_in = demo_actions.unsqueeze(0).to(device)
        demo_masks_in = demo_masks.unsqueeze(0).to(device)
        target_states_in = target_states.unsqueeze(0).to(device)  # (1, T, 19)

        # Predict actions
        with torch.no_grad():
            pred_actions = model(
                demo_states=demo_states_in,
                demo_actions=demo_actions_in,
                demo_mask=demo_masks_in,
                current_states=target_states_in
            )  # (1, T, 4)

        return pred_actions.squeeze(0).cpu()  # (T, 4)

    # Otherwise, process in chunks
    print(f"  Sequence length {T} > {max_seq_len}, processing in chunks...")

    all_pred_actions = []
    chunk_size = max_seq_len

    for start_idx in range(0, T, chunk_size):
        end_idx = min(start_idx + chunk_size, T)
        chunk_states = target_states[start_idx:end_idx]

        # Pad to max_seq_len if needed
        chunk_len = len(chunk_states)
        if chunk_len < max_seq_len:
            padding = torch.zeros(max_seq_len - chunk_len, 19)
            chunk_states_padded = torch.cat([chunk_states, padding], dim=0)
        else:
            chunk_states_padded = chunk_states

        # Prepare input
        demo_states_in = demo_states.unsqueeze(0).to(device)
        demo_actions_in = demo_actions.unsqueeze(0).to(device)
        demo_masks_in = demo_masks.unsqueeze(0).to(device)
        chunk_states_in = chunk_states_padded.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred_chunk = model(
                demo_states=demo_states_in,
                demo_actions=demo_actions_in,
                demo_mask=demo_masks_in,
                current_states=chunk_states_in
            )  # (1, max_seq_len, 4)

        # Take only the valid part (no padding)
        pred_chunk = pred_chunk.squeeze(0)[:chunk_len].cpu()
        all_pred_actions.append(pred_chunk)

    return torch.cat(all_pred_actions, dim=0)  # (T, 4)


def compute_gate_passing(positions, gates, gate_radius=2.0):
    """
    Compute how many gates the drone passed.

    Args:
        positions: (T, 3) trajectory positions
        gates: (N, 3) gate positions
        gate_radius: Radius around gate to count as "passed"

    Returns:
        gates_passed: Number of gates passed
        gate_distances: List of closest distances to each gate
    """
    gates_passed = 0
    gate_distances = []

    for gate in gates:
        # Find minimum distance to this gate across trajectory
        distances = np.linalg.norm(positions - gate, axis=1)
        min_dist = distances.min()
        gate_distances.append(min_dist)

        if min_dist < gate_radius:
            gates_passed += 1

    return gates_passed, gate_distances


def load_full_episode(h5_file, ep_name, dataset):
    """
    Load a FULL episode without max_seq_len truncation.
    This ensures we see the complete trajectory for visualization.
    """
    ep = h5_file[f'episodes/{ep_name}']

    # Get full trajectory length
    T = ep['actions'].shape[0]

    # Subsample (500Hz -> 50Hz)
    subsample_factor = dataset.subsample_factor
    indices = np.arange(0, T, subsample_factor)

    # Load state components (EXACT same order as dataset_icl.py line 252-259)
    state = np.concatenate([
        ep['state/pos'][indices],       # 0-2: position
        ep['state/quat'][indices],      # 3-6: orientation quaternion (w,x,y,z)
        ep['state/lin_vel'][indices],   # 7-9: linear velocity
        ep['state/ang_vel'][indices],   # 10-12: angular velocity
        ep['imu/lin_acc'][indices],     # 13-15: linear acceleration
        ep['imu/ang_vel'][indices]      # 16-18: IMU angular velocity
    ], axis=-1)

    actions = ep['actions'][indices]
    gates = ep['track_gates'][:]

    # Convert to tensors
    state = torch.FloatTensor(state)
    actions = torch.FloatTensor(actions)
    gates = torch.FloatTensor(gates)

    # Normalize
    if dataset.normalize:
        state = (state - dataset.state_mean) / (dataset.state_std + 1e-8)
        actions = (actions - dataset.action_mean) / (dataset.action_std + 1e-8)

    # Create mask (all valid)
    mask = torch.ones(len(state), dtype=torch.bool)

    return {
        'states': state,
        'actions': actions,
        'gates': gates,
        'mask': mask
    }


def create_demo_video(model, dataset, target_ep_name, demo_ep_names, output_path,
                     device='cpu', fps=20, frame_skip=2, demo_color='red',
                     show_gt_trajectory=True):
    """
    Create demo video with ground truth states.

    Args:
        model: ICL policy
        dataset: ICLRacingDataset
        target_ep_name: Target episode name
        demo_ep_names: List of demo episode names
        output_path: Output video path
        device: Device
        fps: Frames per second
        frame_skip: Skip every N frames for faster generation (default 2 = 25Hz instead of 50Hz)
        demo_color: Color for demo trajectories (default 'red')
        show_gt_trajectory: Show ground truth trajectory in addition to predicted (default True)
    """
    print(f"\n{'='*60}")
    print(f"Creating demo video: {target_ep_name}")
    print(f"{'='*60}")

    target_track = dataset.episode_to_track[target_ep_name]

    # Load data - use FULL episodes without truncation
    print("Loading episodes...")
    print(f"  Loading FULL episodes (no max_seq_len truncation)")

    # Load target with full length (for visualization)
    target_data = load_full_episode(dataset.h5_file, target_ep_name, dataset)

    # Load demos using dataset method (respects max_seq_len for model compatibility)
    print(f"  Loading demos with dataset method (respects model's max_seq_len)")
    demo_data = [dataset._load_episode(name) for name in demo_ep_names]

    print(f"  Target episode: {len(target_data['states'])} frames (full length)")
    for i, name in enumerate(demo_ep_names):
        print(f"  Demo {i+1} ({name}): {len(demo_data[i]['states'])} frames (truncated to max_seq_len)")

    # Demos are already padded/truncated by dataset to max_seq_len
    # Just stack them directly
    demo_states = torch.stack([d['states'] for d in demo_data])
    demo_actions = torch.stack([d['actions'] for d in demo_data])
    demo_masks = torch.stack([d['mask'] for d in demo_data])

    # Get target data
    target_states = target_data['states']
    target_actions = target_data['actions']
    target_mask = target_data['mask']
    gates = target_data['gates']

    # Predict actions using ground truth states
    print("Predicting actions...")
    # Get max_seq_len from dataset (matches model training)
    max_seq_len = dataset.max_seq_len
    pred_actions = predict_actions_with_gt_states(
        model, demo_states, demo_actions, demo_masks,
        target_states, device, max_seq_len
    )

    # Denormalize for visualization
    target_states_denorm = dataset.state_mean + target_states * dataset.state_std
    pred_actions_denorm = dataset.action_mean + pred_actions * dataset.action_std
    target_actions_denorm = dataset.action_mean + target_actions * dataset.action_std

    demo_states_denorm = []
    for demo in demo_data:
        demo_states_denorm.append(dataset.state_mean + demo['states'] * dataset.state_std)

    gates_np = gates.numpy()

    # Extract positions
    target_positions = target_states_denorm[:, :3].numpy()  # (T, 3)
    demo_positions = [d[:, :3].numpy() for d in demo_states_denorm]

    # Compute metrics
    valid_len = int(target_mask.sum().item())

    print(f"  Episode length: {len(target_positions)} frames")
    print(f"  Valid length: {valid_len} frames")
    print(f"  Start position: {target_positions[0]}")
    print(f"  End position: {target_positions[valid_len-1]}")

    # Check for movement issues
    total_dist = np.linalg.norm(target_positions[valid_len-1] - target_positions[0])
    print(f"  Total distance (start->end): {total_dist:.2f}m")

    # Check cumulative path length
    path_length = 0.0
    for i in range(1, valid_len):
        path_length += np.linalg.norm(target_positions[i] - target_positions[i-1])
    print(f"  Total path length: {path_length:.2f}m")

    # Check if mostly stationary
    if path_length < 5.0:
        print(f"  ⚠️  WARNING: Episode appears mostly stationary (path < 5m)")
        print(f"  This might be a data issue or the episode is incomplete")

    # Check velocities
    velocities = target_states_denorm[:valid_len, 3:6].numpy()  # Extract velocity (indices 3-5)
    avg_velocity = np.mean(np.linalg.norm(velocities, axis=1))
    max_velocity = np.max(np.linalg.norm(velocities, axis=1))
    print(f"  Avg velocity: {avg_velocity:.2f} m/s, Max velocity: {max_velocity:.2f} m/s")

    # Skip initial hover period (first 10% of trajectory or until drone moves significantly)
    start_idx = 0
    threshold_movement = 0.3  # meters (reduced from 0.5 to catch earlier movement)
    max_hover_check = min(int(valid_len * 0.1), valid_len)  # Only check first 10%

    for i in range(1, max_hover_check):
        movement = np.linalg.norm(target_positions[i] - target_positions[0])
        if movement > threshold_movement:
            start_idx = max(0, i - 10)  # Start a bit earlier to show takeoff
            break

    # If no significant movement detected in first 10%, don't skip anything
    if start_idx == 0 and max_hover_check > 0:
        total_movement = np.linalg.norm(target_positions[valid_len-1] - target_positions[0])
        if total_movement < 1.0:  # Less than 1m total movement
            print(f"  ⚠️  Warning: Very little movement detected (total: {total_movement:.2f}m)")
        print(f"  No hover period detected, using full trajectory")
    else:
        print(f"  Skipping hover period: {start_idx} frames")

    # For metrics, use all frames
    target_positions_valid = target_positions[start_idx:valid_len]
    pred_actions_valid = pred_actions[start_idx:valid_len]
    target_actions_valid = target_actions[start_idx:valid_len]

    # Apply frame skip for faster generation
    # Important: Include the last frame to avoid cutoff
    indices = list(range(start_idx, valid_len, frame_skip))
    if indices[-1] != valid_len - 1:  # Ensure we include the very last frame
        indices.append(valid_len - 1)

    target_positions_subsampled = target_positions[indices]
    pred_actions_subsampled = pred_actions[indices]
    target_actions_subsampled = target_actions[indices]

    print(f"  Total valid frames: {valid_len - start_idx}")
    print(f"  Subsampled frames: {len(indices)} (includes start and end)")

    gates_passed, gate_distances = compute_gate_passing(target_positions_valid, gates_np)

    # Action MSE
    action_mse = torch.mean((pred_actions_valid - target_actions_valid) ** 2).item()

    print(f"\nMetrics:")
    print(f"  Track: {target_track}")
    print(f"  Length: {valid_len - start_idx} steps ({(valid_len - start_idx) * 0.02:.1f}s)")
    print(f"  Gates passed: {gates_passed}/{len(gates_np)}")
    print(f"  Action MSE: {action_mse:.4f}")
    print(f"  Min gate distance: {min(gate_distances):.2f}m")

    # Create video
    print("\nCreating animation...")
    # Use Agg backend for faster rendering without display
    import matplotlib
    matplotlib.use('Agg')

    # Reduce figure size for faster rendering (smaller = faster)
    fig = plt.figure(figsize=(18, 5), dpi=60)  # Lower DPI at creation time

    # Disable interactive mode
    plt.ioff()

    # Create 3 subplots
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    axes = [ax1, ax2, ax3]
    views = [
        (30, 45, "Perspective View"),
        (0, 90, "Top View"),
        (0, 0, "Side View")
    ]

    # Initialize plots
    pred_lines = []  # Predicted (model) trajectories
    gt_lines = []    # Ground truth trajectories
    points = []

    for ax, (elev, azim, view_name) in zip(axes, views):
        # Plot demo trajectories with specified color
        for i, demo_pos in enumerate(demo_positions):
            ax.plot(demo_pos[:, 0], demo_pos[:, 1], demo_pos[:, 2],
                   color=demo_color, alpha=0.5, linewidth=2,
                   label=f'Demo {i+1}' if i < 3 else '',
                   linestyle='--')

        # Plot gates (simplified for faster rendering)
        gate_height = 2.0
        for i, gate in enumerate(gates_np):
            # Simplified gate visualization - just marker, no pole (faster)
            ax.scatter(*gate, c='red', s=100, marker='s', alpha=0.7,
                      edgecolors='darkred', linewidths=1.5)
            ax.text(gate[0], gate[1], gate[2] + 0.5,
                   f'G{i+1}', fontsize=8, fontweight='bold')

        # Initialize predicted trajectory line
        pred_line, = ax.plot([], [], [], 'b-', linewidth=2.5,
                            label='Model Prediction', alpha=0.9)
        pred_lines.append(pred_line)

        # Initialize ground truth trajectory line (if enabled)
        if show_gt_trajectory:
            gt_line, = ax.plot([], [], [], color='green', linewidth=2,
                              label='Ground Truth', alpha=0.6, linestyle=':')
            gt_lines.append(gt_line)

        # Initialize current position marker
        point = ax.scatter([], [], [], c='blue', s=200, marker='o',
                          edgecolors='black', linewidths=2, zorder=5)
        points.append(point)

        # Set view
        ax.view_init(elev=elev, azim=azim)

        # Labels
        ax.set_xlabel('X (m)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
        ax.set_title(view_name, fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)

        # Set limits
        all_data = np.vstack([target_positions] + demo_positions)
        max_range = np.array([all_data[:, 0].max() - all_data[:, 0].min(),
                             all_data[:, 1].max() - all_data[:, 1].min(),
                             all_data[:, 2].max() - all_data[:, 2].min()]).max() / 2.0

        mid_x = (all_data[:, 0].max() + all_data[:, 0].min()) * 0.5
        mid_y = (all_data[:, 1].max() + all_data[:, 1].min()) * 0.5
        mid_z = (all_data[:, 2].max() + all_data[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        if ax == ax1:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Title with metrics (avoid Unicode characters that cause warnings)
    fig.suptitle(
        f'{target_track.capitalize()} Track | '
        f'{len(demo_ep_names)} Demos | '
        f'Gates: {gates_passed}/{len(gates_np)} | '
        f'MSE: {action_mse:.4f} | '
        f'[Validation Set - Never Seen During Training]',
        fontsize=13, fontweight='bold'
    )

    # Animation update function
    def update(frame):
        artists = []

        # Update predicted trajectory lines
        for pred_line in pred_lines:
            pred_line.set_data(target_positions_subsampled[:frame, 0],
                              target_positions_subsampled[:frame, 1])
            pred_line.set_3d_properties(target_positions_subsampled[:frame, 2])
            artists.append(pred_line)

        # Update ground truth trajectory lines (if enabled)
        if show_gt_trajectory:
            for gt_line in gt_lines:
                gt_line.set_data(target_positions_subsampled[:frame, 0],
                                target_positions_subsampled[:frame, 1])
                gt_line.set_3d_properties(target_positions_subsampled[:frame, 2])
                artists.append(gt_line)

        # Update current position marker
        for point in points:
            if frame > 0:
                point._offsets3d = ([target_positions_subsampled[frame-1, 0]],
                                   [target_positions_subsampled[frame-1, 1]],
                                   [target_positions_subsampled[frame-1, 2]])
            artists.append(point)

        return artists

    # Create animation - use subsampled frames
    num_frames = len(target_positions_subsampled)

    # Adaptive frame skip for very long episodes
    if num_frames > 1000:
        print(f"  ⚠️  Episode is very long ({num_frames} frames after frame_skip={frame_skip})")
        print(f"  This will take 5-15 minutes to render!")
        print(f"  Recommendation: Use --frame-skip 4 or higher for faster generation")

        # Further subsample if needed
        if num_frames > 1500:
            extra_skip = 2
            target_positions_subsampled = target_positions_subsampled[::extra_skip]
            num_frames = len(target_positions_subsampled)
            print(f"  Applying additional 2x skip for speed: {num_frames} frames")

    print(f"  Rendering {num_frames} frames...")
    print(f"  Estimated time: {num_frames * 0.15:.0f}-{num_frames * 0.3:.0f} seconds")

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=1000/fps, blit=False,  # blit=False for 3D (faster with 3D plots)
        repeat=False  # Don't repeat (saves memory)
    )

    # Save video
    print(f"Saving video to {output_path}...")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Use MP4 format with ffmpeg for faster generation
    if str(output_path).endswith('.gif'):
        output_path = output_path.with_suffix('.mp4')

    # Try to find ffmpeg
    import shutil
    ffmpeg_path = shutil.which('ffmpeg')

    if ffmpeg_path:
        print(f"  Using ffmpeg from: {ffmpeg_path}")
        try:
            # Set ffmpeg path for matplotlib
            import matplotlib
            matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path

            # Try ffmpeg (much faster)
            writer = animation.FFMpegWriter(
                fps=fps,
                codec='libx264',
                bitrate=1500,
                extra_args=[
                    '-preset', 'ultrafast',
                    '-crf', '30',
                    '-threads', '0',
                    '-tune', 'fastdecode'
                ]
            )
            anim.save(str(output_path), writer=writer, dpi=60)
            print(f"  ✓ Saved with ffmpeg")
        except Exception as e:
            print(f"  ffmpeg failed: {e}")
            print(f"  Falling back to pillow (will take 10-20 minutes)...")
            output_path = output_path.with_suffix('.gif')
            anim.save(str(output_path), writer='pillow', fps=fps, dpi=60)
    else:
        print(f"  ⚠️  ffmpeg not found in PATH")
        print(f"  Falling back to pillow (will take 10-20 minutes)...")
        print(f"  To install ffmpeg: module load ffmpeg (on Zaratan)")
        output_path = output_path.with_suffix('.gif')
        anim.save(str(output_path), writer='pillow', fps=fps, dpi=60)

    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create demo videos with ground truth states')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to policy model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to ORIGINAL dataset HDF5 (not expanded)')
    parser.add_argument('--output-dir', type=str, default='results/demo_videos',
                       help='Output directory')
    parser.add_argument('--num-demos', type=int, default=3,
                       help='Number of demonstrations')
    parser.add_argument('--episodes-per-track', type=int, default=1,
                       help='Number of validation episodes to visualize per track')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for video (default 30 for smooth MP4)')
    parser.add_argument('--frame-skip', type=int, default=2,
                       help='Skip every N frames for faster generation (1=no skip, 2=2x faster)')
    parser.add_argument('--demo-color', type=str, default='red',
                       help='Color for demo trajectories (default: red)')
    parser.add_argument('--show-gt', action='store_true', default=False,
                       help='Show ground truth trajectory in addition to prediction')

    # Custom inference mode arguments
    parser.add_argument('--target-episode', type=str, default=None,
                       help='Specific target episode to test (e.g., ep_000015)')
    parser.add_argument('--demo-episodes', nargs='+', default=None,
                       help='Specific demo episodes to use (e.g., ep_000001 ep_000002 ep_000003)')
    parser.add_argument('--track', type=str, default=None,
                       help='Specific track to test (ellipse, lemniscate, trackRATM)')
    parser.add_argument('--output', type=str, default=None,
                       help='Custom output file path (overrides output-dir)')

    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split (must match training)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (must match training)')

    args = parser.parse_args()

    print("="*60)
    print("ICL Drone Racing - Demo Video Generation")
    print("="*60)
    print(f"Using ground truth states (no dynamics)")
    print(f"Validation episodes only (honest generalization)")
    print(f"Original dataset (full laps)")
    print("="*60)

    # Load original dataset (full laps, not windows)
    print(f"\nLoading dataset: {args.data}")

    # First, check what max_seq_len the model was trained with
    print("\nDetecting model's max_seq_len from checkpoint...")
    checkpoint_temp = torch.load(args.checkpoint, map_location='cpu')
    # Models are typically trained with 256 or 512
    # We'll use 512 as default, but allow full laps to be longer
    model_max_seq_len = 512  # Default from H100 training

    dataset = ICLRacingDataset(
        h5_path=args.data,
        num_demos=args.num_demos,
        max_seq_len=model_max_seq_len,  # Match model training
        use_images=False,
        normalize=True,
        subsample_factor=10
    )

    # Get train/val split (same as training)
    print("\nSplitting into train/val (same as training)...")
    train_episodes, val_episodes = get_validation_split(
        dataset, val_split=args.val_split, seed=args.seed
    )

    print(f"Train episodes: {len(train_episodes)}")
    print(f"Val episodes: {len(val_episodes)}")

    # Group validation episodes by track
    val_by_track = {}
    for ep in val_episodes:
        track = dataset.episode_to_track[ep]
        if track not in val_by_track:
            val_by_track[track] = []
        val_by_track[track].append(ep)

    print(f"\nValidation episodes per track:")
    for track, eps in val_by_track.items():
        print(f"  {track}: {len(eps)} episodes")

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    model = ICLTransformerPolicy(
        state_dim=19,
        action_dim=4,
        embed_dim=256,
        num_heads=8,
        num_decoder_layers=6,
        use_vision=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # ============================================================
    # CUSTOM INFERENCE MODE
    # ============================================================
    if args.target_episode or args.demo_episodes or args.track:
        print(f"\n{'='*60}")
        print("CUSTOM INFERENCE MODE")
        print(f"{'='*60}")

        # Validate target episode
        if args.target_episode:
            if args.target_episode not in dataset.episodes:
                raise ValueError(f"Target episode '{args.target_episode}' not found!")
            target_ep = args.target_episode
            target_track = dataset.episode_to_track[target_ep]
            print(f"Target: {target_ep} (track: {target_track})")
        else:
            # Use first validation episode from specified track
            if args.track:
                track_val_eps = val_by_track.get(args.track, [])
                if not track_val_eps:
                    raise ValueError(f"No validation episodes for track '{args.track}'")
                target_ep = track_val_eps[0]
                target_track = args.track
            else:
                # Use first validation episode overall
                target_ep = val_episodes[0]
                target_track = dataset.episode_to_track[target_ep]
            print(f"Target: {target_ep} (track: {target_track})")

        # Validate demo episodes
        if args.demo_episodes:
            demo_eps = args.demo_episodes
            # Check all exist
            for ep in demo_eps:
                if ep not in dataset.episodes:
                    raise ValueError(f"Demo episode '{ep}' not found!")
                demo_track = dataset.episode_to_track[ep]
                if demo_track != target_track:
                    print(f"⚠️  Warning: Demo {ep} is from {demo_track}, target is {target_track}")
            print(f"Demos: {demo_eps}")
        else:
            # Auto-select demos from same track
            demo_eps = select_demo_episodes(
                dataset, target_ep, train_episodes, val_episodes, args.num_demos
            )
            if len(demo_eps) == 0:
                raise ValueError(f"No validation demos available for {target_ep}")
            print(f"Auto-selected demos: {demo_eps}")

        # Create video
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = output_dir / f"{target_track}_{target_ep}_custom.mp4"

        create_demo_video(
            model, dataset, target_ep, demo_eps, output_path,
            device=args.device, fps=args.fps, frame_skip=args.frame_skip,
            demo_color=args.demo_color, show_gt_trajectory=args.show_gt
        )

        print(f"\n{'='*60}")
        print(f"✅ Custom demo video generated!")
        print(f"Output: {output_path}")
        print(f"{'='*60}")
        return

    # ============================================================
    # BATCH MODE (original behavior)
    # ============================================================
    print(f"\n{'='*60}")
    print("BATCH MODE - Generating Videos for All Tracks")
    print(f"{'='*60}")

    for track, track_val_eps in val_by_track.items():
        print(f"\n{'='*60}")
        print(f"Track: {track.upper()}")
        print(f"{'='*60}")

        # Select episodes to visualize
        num_to_viz = min(args.episodes_per_track, len(track_val_eps))
        selected_eps = np.random.choice(track_val_eps, size=num_to_viz, replace=False)

        for target_ep in selected_eps:
            # Select demo episodes (from validation set only)
            demo_eps = select_demo_episodes(
                dataset, target_ep, train_episodes, val_episodes, args.num_demos
            )

            if len(demo_eps) == 0:
                print(f"⚠️  Skipping {target_ep}: No validation demos available")
                continue

            print(f"\nTarget: {target_ep}")
            print(f"Demos: {demo_eps}")

            # Create video
            output_path = output_dir / f"{track}_{target_ep}_{args.num_demos}demos.mp4"

            create_demo_video(
                model, dataset, target_ep, demo_eps, output_path,
                device=args.device, fps=args.fps, frame_skip=args.frame_skip,
                demo_color=args.demo_color, show_gt_trajectory=args.show_gt
            )

    print(f"\n{'='*60}")
    print("✅ All demo videos generated!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
