"""
Comprehensive evaluation with visualization for ICL drone racing.

Evaluates on validation set and generates:
1. Quantitative metrics (MSE, per-track, per-action)
2. 3D trajectory visualization videos
3. Performance analysis plots
"""

import torch
import numpy as np
import h5py
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import pybullet as p
import pybullet_data
import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_icl.models import ICLTransformerPolicy
from drone_icl.dataset_icl import ICLRacingDataset


def load_model(checkpoint_path, state_dim=26, device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    embed_dim = checkpoint.get('embed_dim', 256)

    model = ICLTransformerPolicy(
        state_dim=state_dim,
        action_dim=4,
        embed_dim=embed_dim
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Handle optional loss values
    train_loss = checkpoint.get('train_loss', None)
    val_loss = checkpoint.get('val_loss', None)

    if train_loss is not None:
        print(f"  Train loss: {train_loss:.4f}")
    else:
        print(f"  Train loss: N/A")

    if val_loss is not None:
        print(f"  Val loss: {val_loss:.4f}")
    else:
        print(f"  Val loss: N/A")

    return model


def evaluate_on_dataset(model, dataset, device='cuda', num_episodes=None):
    """
    Evaluate model on dataset.

    Returns:
        metrics: Dict with overall and per-track metrics
        trajectories: List of (pred, true, gates) for visualization
    """
    model.eval()

    all_metrics = {
        'mse': [],
        'mse_thrust': [],
        'mse_roll': [],
        'mse_pitch': [],
        'mse_yaw': [],
        'track': []
    }

    trajectories = []

    num_eps = num_episodes if num_episodes else len(dataset)

    with torch.no_grad():
        for i in tqdm(range(num_eps), desc="Evaluating"):
            batch = dataset[i]

            # Prepare inputs
            demo_states = batch['demo_states'].unsqueeze(0).to(device)
            demo_actions = batch['demo_actions'].unsqueeze(0).to(device)
            target_states = batch['target_states'].unsqueeze(0).to(device)
            target_actions = batch['target_actions'].unsqueeze(0).to(device)
            demo_masks = batch['demo_masks'].unsqueeze(0).to(device)

            # Print episode info
            if i < 5:  # Print first 5 episodes
                print(f"  Episode {i}: target_length={target_states.shape[1]} steps, track={batch.get('track_type', 'unknown')}")

            # Predict
            pred_actions = model(demo_states, demo_actions, target_states, demo_masks)

            # Compute metrics
            pred = pred_actions[0].cpu()
            true = target_actions[0].cpu()

            mse = torch.mean((pred - true) ** 2).item()
            mse_thrust = torch.mean((pred[:, 0] - true[:, 0]) ** 2).item()
            mse_roll = torch.mean((pred[:, 1] - true[:, 1]) ** 2).item()
            mse_pitch = torch.mean((pred[:, 2] - true[:, 2]) ** 2).item()
            mse_yaw = torch.mean((pred[:, 3] - true[:, 3]) ** 2).item()

            all_metrics['mse'].append(mse)
            all_metrics['mse_thrust'].append(mse_thrust)
            all_metrics['mse_roll'].append(mse_roll)
            all_metrics['mse_pitch'].append(mse_pitch)
            all_metrics['mse_yaw'].append(mse_yaw)
            all_metrics['track'].append(batch.get('track_type', 'unknown'))

            # Save trajectory for visualization (first 10)
            if len(trajectories) < 10:
                # Extract position from state (first 3 dims)
                pred_pos = target_states[0, :, :3].cpu().numpy()
                gates = batch.get('gates', None)
                if gates is not None:
                    gates = gates.cpu().numpy()

                trajectories.append({
                    'pred_actions': pred.numpy(),
                    'true_actions': true.numpy(),
                    'positions': pred_pos,
                    'gates': gates,
                    'track': batch.get('track_type', 'unknown')
                })

    # Aggregate metrics
    metrics = {
        'overall': {
            'mse': np.mean(all_metrics['mse']),
            'mse_std': np.std(all_metrics['mse']),
            'mse_thrust': np.mean(all_metrics['mse_thrust']),
            'mse_roll': np.mean(all_metrics['mse_roll']),
            'mse_pitch': np.mean(all_metrics['mse_pitch']),
            'mse_yaw': np.mean(all_metrics['mse_yaw']),
        }
    }

    # Per-track metrics
    unique_tracks = set(all_metrics['track'])
    metrics['per_track'] = {}

    for track in unique_tracks:
        mask = [t == track for t in all_metrics['track']]
        track_mse = [m for m, msk in zip(all_metrics['mse'], mask) if msk]

        metrics['per_track'][track] = {
            'mse': np.mean(track_mse),
            'mse_std': np.std(track_mse),
            'count': len(track_mse)
        }

    return metrics, trajectories


def visualize_trajectory_3d(trajectory, output_path, fps=10):
    """
    Create 3D animation of drone trajectory with gates.

    Args:
        trajectory: Dict with 'positions', 'gates', 'track'
        output_path: Path to save video
        fps: Frames per second
    """
    positions = trajectory['positions']
    gates = trajectory['gates']
    track = trajectory['track']

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot gates
    if gates is not None:
        ax.scatter(gates[:, 0], gates[:, 1], gates[:, 2],
                  c='red', s=200, marker='o', alpha=0.6, label='Gates')

        # Connect gates to show track
        for i in range(len(gates)):
            next_i = (i + 1) % len(gates)
            ax.plot([gates[i, 0], gates[next_i, 0]],
                   [gates[i, 1], gates[next_i, 1]],
                   [gates[i, 2], gates[next_i, 2]],
                   'r--', alpha=0.3)

    # Initialize trajectory line
    line, = ax.plot([], [], [], 'b-', linewidth=2, label='Drone trajectory')
    point, = ax.plot([], [], [], 'bo', markersize=10)

    # Set limits
    all_pos = positions if gates is None else np.vstack([positions, gates])
    margin = 2.0
    ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    ax.set_zlim(all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Drone Trajectory - Track: {track}')
    ax.legend()

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    def update(frame):
        # Update trajectory up to current frame
        line.set_data(positions[:frame, 0], positions[:frame, 1])
        line.set_3d_properties(positions[:frame, 2])

        # Update current position
        if frame > 0:
            point.set_data([positions[frame-1, 0]], [positions[frame-1, 1]])
            point.set_3d_properties([positions[frame-1, 2]])

        # Rotate view slowly
        ax.view_init(elev=20, azim=frame * 0.5)

        return line, point

    anim = FuncAnimation(fig, update, frames=len(positions),
                        init_func=init, blit=True, interval=1000//fps)

    # Save animation
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='ICL Drone'), bitrate=1800)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"Saved trajectory video to {output_path}")


def plot_metrics(metrics, output_dir):
    """Generate plots for metrics analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Overall MSE breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    actions = ['Overall', 'Thrust', 'Roll', 'Pitch', 'Yaw']
    mse_values = [
        metrics['overall']['mse'],
        metrics['overall']['mse_thrust'],
        metrics['overall']['mse_roll'],
        metrics['overall']['mse_pitch'],
        metrics['overall']['mse_yaw']
    ]

    ax.bar(actions, mse_values, color='skyblue', edgecolor='black')
    ax.set_ylabel('MSE')
    ax.set_title('Action Prediction MSE Breakdown')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mse_breakdown.png', dpi=150)
    plt.close()

    # 2. Per-track performance
    if len(metrics['per_track']) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        tracks = list(metrics['per_track'].keys())
        mse_means = [metrics['per_track'][t]['mse'] for t in tracks]
        mse_stds = [metrics['per_track'][t]['mse_std'] for t in tracks]
        counts = [metrics['per_track'][t]['count'] for t in tracks]

        x = np.arange(len(tracks))
        bars = ax.bar(x, mse_means, yerr=mse_stds, capsize=5,
                     color='lightcoral', edgecolor='black', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(tracks)
        ax.set_ylabel('MSE')
        ax.set_title('Performance by Track Type')
        ax.grid(axis='y', alpha=0.3)

        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={count}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'per_track_performance.png', dpi=150)
        plt.close()

    print(f"Saved plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate and visualize ICL drone racing')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to validation HDF5 dataset')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--num-episodes', type=int, default=None,
                       help='Number of episodes to evaluate (default: all)')
    parser.add_argument('--num-videos', type=int, default=5,
                       help='Number of trajectory videos to generate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--use-gate-info', action='store_true', default=False,
                       help='Use gate information (26D state). Set this if model was trained with --use-gate-info')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print("  ICL Drone Racing - Comprehensive Evaluation")
    print("="*70)
    print()

    # Load model
    state_dim = 26 if args.use_gate_info else 19
    print(f"Loading model (state_dim={state_dim})...")
    model = load_model(args.checkpoint, state_dim=state_dim, device=args.device)
    print()

    # Load dataset
    print(f"Loading validation dataset: {args.data}")
    dataset = ICLRacingDataset(
        h5_path=args.data,
        num_demos=3,
        max_seq_len=512,  # Must be <= 1024 (model's positional encoding limit)
        use_gate_info=args.use_gate_info,
        normalize=True,
        subsample_factor=10  # 500Hz -> 50Hz
    )
    print(f"  Total episodes: {len(dataset)}")
    print()

    # Evaluate
    print("Running evaluation...")
    metrics, trajectories = evaluate_on_dataset(
        model, dataset, device=args.device,
        num_episodes=args.num_episodes
    )
    print()

    # Print results
    print("="*70)
    print("  Evaluation Results")
    print("="*70)
    print()
    print("Overall Performance:")
    print(f"  MSE: {metrics['overall']['mse']:.6f} ± {metrics['overall']['mse_std']:.6f}")
    print(f"  Thrust MSE: {metrics['overall']['mse_thrust']:.6f}")
    print(f"  Roll MSE:   {metrics['overall']['mse_roll']:.6f}")
    print(f"  Pitch MSE:  {metrics['overall']['mse_pitch']:.6f}")
    print(f"  Yaw MSE:    {metrics['overall']['mse_yaw']:.6f}")
    print()

    if metrics['per_track']:
        print("Per-Track Performance:")
        for track, track_metrics in sorted(metrics['per_track'].items()):
            print(f"  {track}:")
            print(f"    MSE: {track_metrics['mse']:.6f} ± {track_metrics['mse_std']:.6f}")
            print(f"    Episodes: {track_metrics['count']}")
        print()

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    print()

    # Generate plots
    print("Generating plots...")
    plot_metrics(metrics, output_dir / 'plots')
    print()

    # Generate trajectory videos
    print(f"Generating {min(args.num_videos, len(trajectories))} trajectory videos...")
    video_dir = output_dir / 'videos'
    video_dir.mkdir(exist_ok=True, parents=True)

    for i, traj in enumerate(trajectories[:args.num_videos]):
        video_path = video_dir / f'trajectory_{i+1}_{traj["track"]}.mp4'
        try:
            visualize_trajectory_3d(traj, video_path, fps=20)
        except Exception as e:
            print(f"Warning: Failed to create video {i+1}: {e}")

    print()
    print("="*70)
    print("  Evaluation Complete!")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
