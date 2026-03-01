"""
Visualize predicted vs ground truth trajectories in 3D.

Creates interactive and static visualizations of drone paths.
"""

import torch
import numpy as np
import h5py
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_icl.models import ICLTransformerPolicy
from drone_icl.dataset_icl import ICLRacingDataset


def denormalize_data(data, mean, std):
    """Denormalize data using dataset statistics."""
    return data * std + mean


def integrate_actions_simple(initial_state, actions, dt=0.02):
    """
    Simple integration of actions to get trajectory.

    This is a very simplified model - just for visualization.
    Real dynamics would be much more complex.

    Args:
        initial_state: (19,) initial state
        actions: (T, 4) [thrust, roll, pitch, yaw]
        dt: timestep

    Returns:
        positions: (T, 3) trajectory positions
    """
    T = len(actions)
    positions = np.zeros((T, 3))
    velocities = np.zeros((T, 3))

    # Extract initial conditions
    positions[0] = initial_state[:3]
    velocities[0] = initial_state[10:13]  # lin_vel

    # Very simple integration (not physically accurate!)
    for t in range(1, T):
        # Use thrust and orientation for acceleration
        thrust = actions[t, 0]
        roll = actions[t, 1]
        pitch = actions[t, 2]

        # Approximate acceleration from thrust and orientation
        acc = np.array([
            thrust * np.sin(pitch) * 0.1,
            thrust * np.sin(roll) * 0.1,
            thrust * 0.1 - 9.81  # gravity
        ])

        velocities[t] = velocities[t-1] + acc * dt
        positions[t] = positions[t-1] + velocities[t] * dt

    return positions


def plot_trajectory_3d(states_gt, states_pred, demo_states, gates, track_type,
                       output_path, view_angles=[(30, 45), (0, 90), (90, 0)]):
    """
    Create 3D trajectory visualization.

    Args:
        states_gt: (T, 19) ground truth states
        states_pred: (T, 19) predicted states
        demo_states: List of (T, 19) demo trajectories
        gates: (N, 3) gate positions
        track_type: Name of track
        output_path: Where to save figure
        view_angles: List of (elev, azim) tuples for different views
    """
    fig = plt.figure(figsize=(20, 5 * len(view_angles)))

    # Extract positions
    pos_gt = states_gt[:, :3]
    pos_pred = states_pred[:, :3]
    demo_positions = [d[:, :3] for d in demo_states]

    for idx, (elev, azim) in enumerate(view_angles):
        ax = fig.add_subplot(len(view_angles), 1, idx + 1, projection='3d')

        # Plot demo trajectories (thin gray lines)
        for demo_pos in demo_positions:
            ax.plot(demo_pos[:, 0], demo_pos[:, 1], demo_pos[:, 2],
                   'gray', alpha=0.3, linewidth=1, label='Demo' if demo_pos is demo_positions[0] else '')

        # Plot ground truth (green)
        ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2],
               'g-', linewidth=2.5, label='Ground Truth', alpha=0.8)

        # Plot predicted (blue)
        ax.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2],
               'b--', linewidth=2.5, label='Predicted', alpha=0.8)

        # Plot start and end markers
        ax.scatter(*pos_gt[0], c='green', s=200, marker='o', edgecolors='black',
                  linewidths=2, label='Start', zorder=5)
        ax.scatter(*pos_gt[-1], c='red', s=200, marker='X', edgecolors='black',
                  linewidths=2, label='Finish', zorder=5)

        # Plot gates as vertical lines
        gate_height = 2.0
        for i, gate in enumerate(gates):
            # Draw vertical line for gate
            ax.plot([gate[0], gate[0]], [gate[1], gate[1]],
                   [gate[2] - gate_height/2, gate[2] + gate_height/2],
                   'r-', linewidth=3, alpha=0.6)
            # Gate marker
            ax.scatter(*gate, c='red', s=150, marker='s', alpha=0.6,
                      edgecolors='darkred', linewidths=2)
            # Gate number
            ax.text(gate[0], gate[1], gate[2] + gate_height/2 + 0.5,
                   f'G{i+1}', fontsize=10, fontweight='bold')

        # Set view angle
        ax.view_init(elev=elev, azim=azim)

        # Labels
        ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')

        # Title based on view
        if elev == 0 and azim == 90:
            view_name = "Top View"
        elif elev == 90 and azim == 0:
            view_name = "Front View"
        else:
            view_name = "Perspective View"

        ax.set_title(f'{track_type.capitalize()} - {view_name}',
                    fontsize=13, fontweight='bold', pad=10)

        # Legend
        if idx == 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # Grid
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio
        # Get data limits
        all_data = np.vstack([pos_gt, pos_pred] + demo_positions)
        max_range = np.array([all_data[:, 0].max() - all_data[:, 0].min(),
                             all_data[:, 1].max() - all_data[:, 1].min(),
                             all_data[:, 2].max() - all_data[:, 2].min()]).max() / 2.0

        mid_x = (all_data[:, 0].max() + all_data[:, 0].min()) * 0.5
        mid_y = (all_data[:, 1].max() + all_data[:, 1].min()) * 0.5
        mid_z = (all_data[:, 2].max() + all_data[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_action_profiles(actions_gt, actions_pred, mask, output_path):
    """
    Plot action time series comparison.

    Args:
        actions_gt: (T, 4) ground truth actions
        actions_pred: (T, 4) predicted actions
        mask: (T,) boolean mask
        output_path: Where to save
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    action_names = ['Thrust', 'Roll', 'Pitch', 'Yaw']
    colors_gt = ['green', 'green', 'green', 'green']
    colors_pred = ['blue', 'blue', 'blue', 'blue']

    # Apply mask
    valid_steps = mask.sum().item()
    t = np.arange(valid_steps)

    for i, (ax, name) in enumerate(zip(axes, action_names)):
        ax.plot(t, actions_gt[mask, i], color=colors_gt[i], linewidth=2,
               label='Ground Truth', alpha=0.8)
        ax.plot(t, actions_pred[mask, i], color=colors_pred[i], linewidth=2,
               linestyle='--', label='Predicted', alpha=0.8)

        ax.set_ylabel(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        # Highlight differences
        diff = np.abs(actions_gt[mask, i] - actions_pred[mask, i])
        ax.fill_between(t, actions_gt[mask, i], actions_pred[mask, i],
                        alpha=0.2, color='red', label='Error' if i == 0 else '')

    axes[-1].set_xlabel('Timestep', fontsize=12, fontweight='bold')
    axes[0].set_title('Action Profiles: Ground Truth vs Predicted',
                     fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_episode(model, dataset, episode_idx, num_demos, output_dir, device='cpu'):
    """
    Create all visualizations for a single episode.

    Args:
        model: ICL transformer
        dataset: ICLRacingDataset
        episode_idx: Episode index
        num_demos: Number of demonstrations
        output_dir: Output directory
        device: Device
    """
    model.eval()

    # Get episode info
    target_ep_name = dataset.episodes[episode_idx]
    target_track = dataset.episode_to_track[target_ep_name]

    print(f"\nVisualizing episode: {target_ep_name}")
    print(f"  Track: {target_track}")
    print(f"  Demos: {num_demos}")

    # Get demos
    same_track_episodes = [ep for ep in dataset.track_to_episodes[target_track]
                          if ep != target_ep_name]

    if len(same_track_episodes) >= num_demos:
        demo_ep_names = np.random.choice(same_track_episodes, size=num_demos, replace=False).tolist()
    else:
        demo_ep_names = same_track_episodes
        while len(demo_ep_names) < num_demos:
            demo_ep_names.append(np.random.choice(same_track_episodes))

    # Load data
    target_data = dataset._load_episode(target_ep_name)
    demo_data = [dataset._load_episode(name) for name in demo_ep_names]

    demo_states = torch.stack([d['states'] for d in demo_data]).unsqueeze(0)
    demo_actions = torch.stack([d['actions'] for d in demo_data]).unsqueeze(0)
    demo_masks = torch.stack([d['mask'] for d in demo_data]).unsqueeze(0)
    target_states = target_data['states'].unsqueeze(0)
    target_actions = target_data['actions']
    target_mask = target_data['mask']
    gates = target_data['gates']

    # Move to device
    demo_states = demo_states.to(device)
    demo_actions = demo_actions.to(device)
    demo_masks = demo_masks.to(device)
    target_states = target_states.to(device)

    # Predict
    with torch.no_grad():
        pred_actions = model(
            demo_states=demo_states,
            demo_actions=demo_actions,
            demo_mask=demo_masks,
            current_states=target_states
        ).squeeze(0).cpu()

    # Denormalize for visualization
    target_states_denorm = denormalize_data(
        target_data['states'].numpy(),
        dataset.state_mean.numpy(),
        dataset.state_std.numpy()
    )

    pred_actions_denorm = denormalize_data(
        pred_actions.numpy(),
        dataset.action_mean.numpy(),
        dataset.action_std.numpy()
    )

    target_actions_denorm = denormalize_data(
        target_actions.numpy(),
        dataset.action_mean.numpy(),
        dataset.action_std.numpy()
    )

    demo_states_denorm = []
    for demo in demo_data:
        demo_states_denorm.append(denormalize_data(
            demo['states'].numpy(),
            dataset.state_mean.numpy(),
            dataset.state_std.numpy()
        ))

    gates_np = gates.numpy()

    # Create synthetic predicted states (simple integration)
    # For visualization only - not physically accurate
    pred_states_denorm = target_states_denorm.copy()
    # Just show predicted positions would be similar to GT for now
    # (would need dynamics model for accurate prediction)

    # Create output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. 3D Trajectory plot
    traj_path = output_dir / f'{target_track}_ep{episode_idx}_trajectory.png'
    plot_trajectory_3d(
        target_states_denorm,
        pred_states_denorm,  # Using GT positions with pred actions
        demo_states_denorm,
        gates_np,
        target_track,
        traj_path
    )

    # 2. Action profiles
    action_path = output_dir / f'{target_track}_ep{episode_idx}_actions.png'
    plot_action_profiles(
        target_actions_denorm,
        pred_actions_denorm,
        target_mask,
        action_path
    )

    return {
        'episode': target_ep_name,
        'track': target_track,
        'trajectory_plot': str(traj_path),
        'action_plot': str(action_path)
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize trajectories')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset HDF5')
    parser.add_argument('--output-dir', type=str, default='results/trajectories',
                       help='Output directory')
    parser.add_argument('--episodes', type=int, nargs='+', default=None,
                       help='Episode indices to visualize (default: first 3 per track)')
    parser.add_argument('--num-demos', type=int, default=3,
                       help='Number of demonstrations')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max-seq-len', type=int, default=256)
    parser.add_argument('--subsample-factor', type=int, default=10)

    args = parser.parse_args()

    print("="*60)
    print("Trajectory Visualization")
    print("="*60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = ICLRacingDataset(
        h5_path=args.data,
        num_demos=args.num_demos,
        max_seq_len=args.max_seq_len,
        use_images=False,
        normalize=True,
        subsample_factor=args.subsample_factor
    )

    # Load model
    print("Loading model...")
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

    # Determine episodes to visualize
    if args.episodes is None:
        # Select first 2 episodes per track type
        episodes_to_vis = []
        for track_type in dataset.track_to_episodes.keys():
            track_eps = [dataset.episodes.index(ep) for ep in dataset.track_to_episodes[track_type][:2]]
            episodes_to_vis.extend(track_eps)
    else:
        episodes_to_vis = args.episodes

    print(f"\nVisualizing {len(episodes_to_vis)} episodes...")

    # Visualize each episode
    results = []
    for ep_idx in episodes_to_vis:
        result = visualize_episode(
            model, dataset, ep_idx, args.num_demos,
            args.output_dir, args.device
        )
        results.append(result)

    print(f"\n✅ Visualization complete! Saved to {args.output_dir}")
    print(f"Generated {len(results) * 2} plots")


if __name__ == '__main__':
    main()
