"""
Autoregressive trajectory rollout with learned dynamics.

This generates complete trajectories step-by-step, showing the drone "flying"
using the predicted actions.
"""

import torch
import torch.nn as nn
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


class SimpleDynamicsModel(nn.Module):
    """
    Simple learned forward dynamics: state_t, action_t -> state_t+1

    This is a simplified model for demonstration.
    Real drone dynamics would be much more complex.
    """

    def __init__(self, state_dim=19, action_dim=4, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        """
        Predict next state.

        Args:
            state: (B, state_dim)
            action: (B, action_dim)

        Returns:
            next_state: (B, state_dim)
        """
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        return state + delta  # Residual connection


def train_dynamics_model(dataset, epochs=50, batch_size=128, device='cpu'):
    """
    Train a simple forward dynamics model on the dataset.

    Args:
        dataset: ICLRacingDataset
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on

    Returns:
        Trained dynamics model
    """
    print("\nTraining dynamics model...")

    model = SimpleDynamicsModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Collect all state-action-next_state pairs
    print("Collecting training data...")
    states = []
    actions_data = []
    next_states = []

    for ep_idx in tqdm(range(len(dataset)), desc="Loading episodes"):
        ep_data = dataset._load_episode(dataset.episodes[ep_idx])
        s = ep_data['states']
        a = ep_data['actions']
        m = ep_data['mask']

        # Get valid transitions
        valid_len = m.sum().item()
        if valid_len > 1:
            states.append(s[:valid_len-1])
            actions_data.append(a[:valid_len-1])
            next_states.append(s[1:valid_len])

    states = torch.cat(states, dim=0)
    actions_data = torch.cat(actions_data, dim=0)
    next_states = torch.cat(next_states, dim=0)

    print(f"Collected {len(states)} transitions")

    # Create dataloader
    train_dataset = torch.utils.data.TensorDataset(states, actions_data, next_states)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for s, a, ns in train_loader:
            s, a, ns = s.to(device), a.to(device), ns.to(device)

            optimizer.zero_grad()
            pred_ns = model(s, a)
            loss = criterion(pred_ns, ns)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    print("✅ Dynamics model trained!")
    model.eval()
    return model


def rollout_trajectory(policy_model, dynamics_model, demo_states, demo_actions,
                      demo_masks, initial_state, gates, max_steps=300, device='cpu'):
    """
    Generate trajectory autoregressively.

    Args:
        policy_model: ICL transformer
        dynamics_model: Forward dynamics model
        demo_states: (N, T, state_dim) demonstration states
        demo_actions: (N, T, action_dim) demonstration actions
        demo_masks: (N, T) demonstration masks
        initial_state: (state_dim,) initial state
        gates: (num_gates, 3) gate positions
        max_steps: Maximum rollout steps
        device: Device

    Returns:
        rollout_states: (T, state_dim) generated trajectory
        rollout_actions: (T, action_dim) generated actions
        metrics: Dictionary with rollout metrics
    """
    policy_model.eval()
    dynamics_model.eval()

    rollout_states = [initial_state.clone()]
    rollout_actions = []

    # Prepare demo context
    demo_states = demo_states.unsqueeze(0).to(device)  # (1, N, T, D)
    demo_actions = demo_actions.unsqueeze(0).to(device)
    demo_masks = demo_masks.unsqueeze(0).to(device)

    current_state = initial_state.unsqueeze(0).to(device)  # (1, D)

    # Rollout loop
    with torch.no_grad():
        for step in range(max_steps):
            # Prepare query (single timestep with padding)
            query_states = torch.zeros(1, max_steps, 19).to(device)
            query_states[0, step] = current_state.squeeze(0)

            # Predict action
            pred_actions = policy_model(
                demo_states=demo_states,
                demo_actions=demo_actions,
                demo_mask=demo_masks,
                current_states=query_states
            )  # (1, T, 4)

            pred_action = pred_actions[0, step]  # (4,)

            # Apply action through dynamics
            next_state = dynamics_model(
                current_state,
                pred_action.unsqueeze(0)
            ).squeeze(0)  # (D,)

            # Store
            rollout_actions.append(pred_action.cpu())
            rollout_states.append(next_state.cpu())

            # Update current state
            current_state = next_state.unsqueeze(0)

            # Check termination (e.g., if drone crashes or goes too far)
            pos = next_state[:3].cpu().numpy()
            if np.abs(pos).max() > 200:  # Out of bounds
                print(f"Rollout terminated at step {step}: out of bounds")
                break

    rollout_states = torch.stack(rollout_states[:-1])  # Exclude last state
    rollout_actions = torch.stack(rollout_actions)

    # Compute metrics
    rollout_positions = rollout_states[:, :3].numpy()

    # Count gates passed (simple distance check)
    gates_passed = 0
    gate_threshold = 2.0  # meters
    gates_np = gates.cpu().numpy() if isinstance(gates, torch.Tensor) else gates
    for gate in gates_np:
        distances = np.linalg.norm(rollout_positions - gate, axis=1)
        if distances.min() < gate_threshold:
            gates_passed += 1

    metrics = {
        'total_steps': len(rollout_states),
        'gates_passed': gates_passed,
        'total_gates': len(gates),
        'completion_rate': gates_passed / len(gates),
        'final_position': rollout_positions[-1].tolist(),
        'max_position': np.abs(rollout_positions).max()
    }

    return rollout_states, rollout_actions, metrics


def create_rollout_video(rollout_states, demo_states_list, gates, track_type,
                         output_path, fps=20):
    """
    Create animated video of rollout trajectory.

    Args:
        rollout_states: (T, state_dim) generated trajectory
        demo_states_list: List of (T, state_dim) demo trajectories
        gates: (N, 3) gate positions
        track_type: Track name
        output_path: Output video path
        fps: Frames per second
    """
    fig = plt.figure(figsize=(16, 6))

    # Create 3 subplots: perspective, top view, side view
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    axes = [ax1, ax2, ax3]
    views = [
        (30, 45, "Perspective"),
        (0, 90, "Top View"),
        (90, 0, "Side View")
    ]

    # Extract positions
    rollout_pos = rollout_states[:, :3].numpy()
    demo_positions = [d[:, :3] for d in demo_states_list]

    # Initialize lines and points
    lines = []
    points = []

    for ax, (elev, azim, view_name) in zip(axes, views):
        # Plot demo trajectories
        for demo_pos in demo_positions:
            ax.plot(demo_pos[:, 0], demo_pos[:, 1], demo_pos[:, 2],
                   'gray', alpha=0.3, linewidth=1)

        # Plot gates
        gate_height = 2.0
        for gate in gates:
            ax.plot([gate[0], gate[0]], [gate[1], gate[1]],
                   [gate[2] - gate_height/2, gate[2] + gate_height/2],
                   'r-', linewidth=2, alpha=0.5)
            ax.scatter(*gate, c='red', s=100, marker='s', alpha=0.5)

        # Initialize rollout line and current position
        line, = ax.plot([], [], [], 'b-', linewidth=2.5, label='Rollout')
        point = ax.scatter([], [], [], c='blue', s=150, marker='o',
                          edgecolors='black', linewidths=2)

        lines.append(line)
        points.append(point)

        # Set view
        ax.view_init(elev=elev, azim=azim)

        # Labels
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_zlabel('Z (m)', fontsize=9)
        ax.set_title(f'{view_name}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Set limits
        all_data = np.vstack([rollout_pos] + demo_positions)
        max_range = np.array([all_data[:, 0].max() - all_data[:, 0].min(),
                             all_data[:, 1].max() - all_data[:, 1].min(),
                             all_data[:, 2].max() - all_data[:, 2].min()]).max() / 2.0

        mid_x = (all_data[:, 0].max() + all_data[:, 0].min()) * 0.5
        mid_y = (all_data[:, 1].max() + all_data[:, 1].min()) * 0.5
        mid_z = (all_data[:, 2].max() + all_data[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    fig.suptitle(f'Autoregressive Rollout: {track_type.capitalize()}',
                fontsize=14, fontweight='bold')

    # Animation update function
    def update(frame):
        for line, point in zip(lines, points):
            # Update trajectory line
            line.set_data(rollout_pos[:frame, 0], rollout_pos[:frame, 1])
            line.set_3d_properties(rollout_pos[:frame, 2])

            # Update current position
            if frame > 0:
                point._offsets3d = ([rollout_pos[frame-1, 0]],
                                   [rollout_pos[frame-1, 1]],
                                   [rollout_pos[frame-1, 2]])

        return lines + points

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(rollout_pos),
        interval=1000/fps, blit=False
    )

    # Save
    print(f"Rendering video... (this may take a few minutes)")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=1800)
    anim.save(output_path, writer=writer)

    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Autoregressive trajectory rollout')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to policy model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset HDF5')
    parser.add_argument('--output-dir', type=str, default='results/rollouts',
                       help='Output directory')
    parser.add_argument('--episode-idx', type=int, default=0,
                       help='Episode index to rollout')
    parser.add_argument('--num-demos', type=int, default=3,
                       help='Number of demonstrations')
    parser.add_argument('--max-steps', type=int, default=300,
                       help='Maximum rollout steps')
    parser.add_argument('--train-dynamics', action='store_true',
                       help='Train dynamics model (otherwise loads from file)')
    parser.add_argument('--dynamics-path', type=str, default='checkpoints/dynamics_model.pt',
                       help='Path to dynamics model')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max-seq-len', type=int, default=256)
    parser.add_argument('--subsample-factor', type=int, default=10)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("Autoregressive Trajectory Rollout")
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

    # Load policy model
    print("Loading policy model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    policy_model = ICLTransformerPolicy(
        state_dim=19,
        action_dim=4,
        embed_dim=256,
        num_heads=8,
        num_decoder_layers=6,
        use_vision=False
    )

    policy_model.load_state_dict(checkpoint['model_state_dict'])
    policy_model.to(args.device)
    policy_model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Load or train dynamics model
    dynamics_path = Path(args.dynamics_path)
    if args.train_dynamics or not dynamics_path.exists():
        dynamics_model = train_dynamics_model(dataset, epochs=50, device=args.device)
        # Save dynamics model
        dynamics_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(dynamics_model.state_dict(), dynamics_path)
        print(f"Saved dynamics model to {dynamics_path}")
    else:
        print(f"Loading dynamics model from {dynamics_path}")
        dynamics_model = SimpleDynamicsModel().to(args.device)
        dynamics_model.load_state_dict(torch.load(dynamics_path, map_location=args.device))
        dynamics_model.eval()

    # Get episode to rollout
    target_ep_name = dataset.episodes[args.episode_idx]
    target_track = dataset.episode_to_track[target_ep_name]

    print(f"\nRolling out episode: {target_ep_name}")
    print(f"  Track: {target_track}")

    # Get demos
    same_track_episodes = [ep for ep in dataset.track_to_episodes[target_track]
                          if ep != target_ep_name]

    demo_ep_names = np.random.choice(same_track_episodes, size=min(args.num_demos, len(same_track_episodes)),
                                     replace=False).tolist()

    # Load data
    target_data = dataset._load_episode(target_ep_name)
    demo_data = [dataset._load_episode(name) for name in demo_ep_names]

    demo_states = torch.stack([d['states'] for d in demo_data])
    demo_actions = torch.stack([d['actions'] for d in demo_data])
    demo_masks = torch.stack([d['mask'] for d in demo_data])
    initial_state = target_data['states'][0]
    gates = target_data['gates']

    # Perform rollout
    print(f"\nPerforming rollout (max {args.max_steps} steps)...")
    rollout_states, rollout_actions, metrics = rollout_trajectory(
        policy_model, dynamics_model,
        demo_states, demo_actions, demo_masks,
        initial_state, gates,
        max_steps=args.max_steps,
        device=args.device
    )

    print(f"\n Rollout Metrics:")
    print(f"  Steps: {metrics['total_steps']}")
    print(f"  Gates passed: {metrics['gates_passed']}/{metrics['total_gates']}")
    print(f"  Completion rate: {metrics['completion_rate']*100:.1f}%")

    # Denormalize for visualization
    def denorm(data, mean, std):
        return data * std + mean

    rollout_states_denorm = denorm(rollout_states, dataset.state_mean, dataset.state_std)
    demo_states_denorm = [denorm(d['states'], dataset.state_mean, dataset.state_std) for d in demo_data]

    # Create video
    video_path = output_dir / f'{target_track}_ep{args.episode_idx}_rollout.mp4'
    print(f"\nCreating rollout video...")

    try:
        create_rollout_video(
            rollout_states_denorm,
            demo_states_denorm,
            gates.numpy(),
            target_track,
            video_path,
            fps=20
        )
    except Exception as e:
        print(f"⚠️  Could not create video: {e}")
        print("    (ffmpeg may not be installed)")

        # Create static plot instead
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        rollout_pos = rollout_states_denorm[:, :3].numpy()

        for demo in demo_states_denorm:
            demo_pos = demo[:, :3].numpy()
            ax.plot(demo_pos[:, 0], demo_pos[:, 1], demo_pos[:, 2],
                   'gray', alpha=0.3, linewidth=1)

        ax.plot(rollout_pos[:, 0], rollout_pos[:, 1], rollout_pos[:, 2],
               'b-', linewidth=2.5, label='Rollout')

        for i, gate in enumerate(gates.numpy()):
            ax.scatter(*gate, c='red', s=150, marker='s')

        ax.set_title(f'Rollout: {target_track}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()

        static_path = output_dir / f'{target_track}_ep{args.episode_idx}_rollout_static.png'
        plt.savefig(static_path, dpi=300)
        print(f"Saved static plot: {static_path}")

    print(f"\n✅ Rollout complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
