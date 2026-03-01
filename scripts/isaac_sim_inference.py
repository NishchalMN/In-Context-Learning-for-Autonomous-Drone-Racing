"""
Isaac Sim Real-time Inference Script

Connects trained ICL model to Isaac Sim drone for real-time control.
Runs in 3rd person view with gates from track configuration.

Usage:
    python isaac_sim_inference.py \
        --checkpoint path/to/model.pt \
        --track-config path/to/track_gates.json \
        --demo-episodes ep_000001 ep_000002 ep_000003 \
        --device cuda
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_icl.model_icl import ICLTransformer


class IsaacSimICLController:
    """
    Real-time ICL controller for Isaac Sim drone.
    """

    def __init__(self, model, demo_states, demo_actions, demo_masks,
                 state_mean, state_std, action_mean, action_std,
                 device='cuda', max_seq_len=512):
        """
        Initialize controller.

        Args:
            model: Trained ICL policy
            demo_states: (N, T_demo, 19) demonstration states (normalized)
            demo_actions: (N, T_demo, 4) demonstration actions (normalized)
            demo_masks: (N, T_demo) demonstration masks
            state_mean: State normalization mean
            state_std: State normalization std
            action_mean: Action normalization mean
            action_std: Action normalization std
            device: Device
            max_seq_len: Maximum sequence length
        """
        self.model = model.to(device)
        self.model.eval()

        self.demo_states = demo_states.to(device)
        self.demo_actions = demo_actions.to(device)
        self.demo_masks = demo_masks.to(device)

        self.state_mean = state_mean.to(device)
        self.state_std = state_std.to(device)
        self.action_mean = action_mean.to(device)
        self.action_std = action_std.to(device)

        self.device = device
        self.max_seq_len = max_seq_len

        # History buffer for current episode
        self.state_history = []
        self.action_history = []

        print(f"✅ Controller initialized with {len(demo_states)} demos")
        print(f"   Device: {device}")
        print(f"   Max sequence length: {max_seq_len}")

    def reset(self):
        """Reset controller for new episode."""
        self.state_history = []
        self.action_history = []
        print("🔄 Controller reset")

    def predict_action(self, current_state):
        """
        Predict action for current state.

        Args:
            current_state: (19,) current drone state (denormalized)
                [pos_x, pos_y, pos_z,           # 0-2: position
                 vel_x, vel_y, vel_z,           # 3-5: velocity
                 quat_w, quat_x, quat_y, quat_z,  # 6-9: orientation
                 ang_vel_x, ang_vel_y, ang_vel_z,  # 10-12: angular velocity
                 gate_x, gate_y, gate_z,        # 13-15: next gate position
                 gate_dist, gate_angle, gate_idx]  # 16-18: gate metrics

        Returns:
            action: (4,) predicted action [thrust, roll, pitch, yaw] (denormalized)
        """
        # Normalize current state
        state_tensor = torch.from_numpy(current_state).float()
        state_norm = (state_tensor - self.state_mean) / (self.state_std + 1e-8)

        # Add to history
        self.state_history.append(state_norm)

        # Keep only last max_seq_len states
        if len(self.state_history) > self.max_seq_len:
            self.state_history = self.state_history[-self.max_seq_len:]

        # Prepare input
        current_states = torch.stack(self.state_history).unsqueeze(0).to(self.device)  # (1, T, 19)
        demo_states_in = self.demo_states.unsqueeze(0)  # (1, N, T_demo, 19)
        demo_actions_in = self.demo_actions.unsqueeze(0)  # (1, N, T_demo, 4)
        demo_masks_in = self.demo_masks.unsqueeze(0)  # (1, N, T_demo)

        # Predict
        with torch.no_grad():
            pred_actions_norm = self.model(
                demo_states=demo_states_in,
                demo_actions=demo_actions_in,
                demo_mask=demo_masks_in,
                current_states=current_states
            )  # (1, T, 4)

        # Get last action and denormalize
        last_action_norm = pred_actions_norm[0, -1].cpu()
        action = (last_action_norm * self.action_std + self.action_mean).numpy()

        # Clip to safe ranges
        action[0] = np.clip(action[0], 0.0, 1.0)     # thrust [0, 1]
        action[1:] = np.clip(action[1:], -1.0, 1.0)  # roll, pitch, yaw [-1, 1]

        # Store action in history (for potential future use)
        self.action_history.append(last_action_norm)

        return action

    def get_state_from_isaac(self, drone_pose, drone_velocity, drone_angular_velocity,
                            gates, current_gate_idx):
        """
        Construct state vector from Isaac Sim observations.

        Args:
            drone_pose: (7,) [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
            drone_velocity: (3,) [vel_x, vel_y, vel_z]
            drone_angular_velocity: (3,) [ang_vel_x, ang_vel_y, ang_vel_z]
            gates: (N, 3) array of gate positions
            current_gate_idx: int, index of next gate to pass

        Returns:
            state: (19,) state vector
        """
        # Extract pose components
        pos = drone_pose[:3]
        quat = drone_pose[3:]  # [w, x, y, z]

        # Get next gate
        if current_gate_idx < len(gates):
            next_gate = gates[current_gate_idx]
        else:
            next_gate = gates[-1]  # Use last gate if all passed

        # Compute gate metrics
        gate_vec = next_gate - pos
        gate_dist = np.linalg.norm(gate_vec)

        # Gate angle (simplified - angle between forward direction and gate vector)
        # For more accurate angle, compute using drone orientation
        forward = np.array([1.0, 0.0, 0.0])  # Simplified forward direction
        gate_angle = np.arctan2(gate_vec[1], gate_vec[0])

        # Construct state
        state = np.concatenate([
            pos,                    # 0-2: position
            drone_velocity,         # 3-5: velocity
            quat,                   # 6-9: orientation
            drone_angular_velocity, # 10-12: angular velocity
            next_gate,              # 13-15: next gate position
            [gate_dist],            # 16: gate distance
            [gate_angle],           # 17: gate angle
            [float(current_gate_idx)]  # 18: gate index
        ])

        return state


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load model checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config
    config = checkpoint.get('config', {
        'state_dim': 19,
        'action_dim': 4,
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 4,
        'max_demos': 5,
        'max_seq_len': 512,
        'dropout': 0.1
    })

    # Create model
    model = ICLTransformer(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    return model, config


def load_demo_episodes(demo_ep_names, dataset):
    """Load demonstration episodes from dataset."""
    print(f"\nLoading {len(demo_ep_names)} demo episodes...")

    demo_states = []
    demo_actions = []
    demo_masks = []

    # Verify all demos exist and are from same track
    tracks = set()
    for ep_name in demo_ep_names:
        if ep_name not in dataset.episodes:
            raise ValueError(f"Demo episode '{ep_name}' not found in dataset!")

        track = dataset.episode_to_track.get(ep_name, 'unknown')
        tracks.add(track)

        ep_data = dataset._load_episode(ep_name)
        demo_states.append(ep_data['states'])
        demo_actions.append(ep_data['actions'])
        demo_masks.append(ep_data['mask'])
        print(f"  ✅ {ep_name} (track: {track})")

    if len(tracks) > 1:
        print(f"  ⚠️  Warning: Demos are from different tracks: {tracks}")
        print(f"     Best performance when all demos are from the same track type")

    demo_states = torch.stack(demo_states)
    demo_actions = torch.stack(demo_actions)
    demo_masks = torch.stack(demo_masks)

    return demo_states, demo_actions, demo_masks, list(tracks)[0]


def load_track_gates(track_config_path):
    """
    Load gate positions from track configuration.

    Expected JSON format:
    {
        "track_name": "ellipse",
        "gates": [
            {"id": 0, "position": [x, y, z]},
            {"id": 1, "position": [x, y, z]},
            ...
        ]
    }
    """
    print(f"\nLoading track configuration: {track_config_path}")
    with open(track_config_path, 'r') as f:
        config = json.load(f)

    track_name = config.get('track_name', 'unknown')
    gates = []

    for gate in config['gates']:
        gates.append(gate['position'])

    gates = np.array(gates, dtype=np.float32)

    print(f"✅ Loaded {len(gates)} gates for track: {track_name}")
    return gates, track_name


def main():
    parser = argparse.ArgumentParser(description='Isaac Sim real-time inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset HDF5 (for loading demos)')
    parser.add_argument('--track-config', type=str, required=True,
                       help='Path to track gate configuration JSON')
    parser.add_argument('--demo-episodes', nargs='+', required=True,
                       help='List of demo episode names (e.g., ep_000001 ep_000002)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--control-freq', type=int, default=50,
                       help='Control frequency (Hz)')

    args = parser.parse_args()

    print("="*60)
    print("Isaac Sim ICL Real-time Inference")
    print("="*60)

    # Load model
    model, config = load_checkpoint(args.checkpoint, args.device)

    # Load dataset for normalization stats and demos
    from drone_icl.dataset_icl import ICLRacingDataset

    print(f"\nLoading dataset: {args.data}")
    dataset = ICLRacingDataset(
        h5_path=args.data,
        num_demos=len(args.demo_episodes),
        max_seq_len=config['max_seq_len'],
        use_images=False,
        normalize=True,
        subsample_factor=10
    )

    # Load demo episodes
    demo_states, demo_actions, demo_masks, demo_track = load_demo_episodes(
        args.demo_episodes, dataset
    )

    # Load track gates
    gates, track_name = load_track_gates(args.track_config)

    # Validate track alignment
    if demo_track != track_name:
        print(f"\n⚠️  WARNING: Track mismatch!")
        print(f"   Demo episodes are from: {demo_track}")
        print(f"   Track config is for: {track_name}")
        print(f"   Model may not perform well with mismatched tracks!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return None, None
    else:
        print(f"\n✅ Track alignment verified: {track_name}")

    # Create controller
    controller = IsaacSimICLController(
        model=model,
        demo_states=demo_states,
        demo_actions=demo_actions,
        demo_masks=demo_masks,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        action_mean=dataset.action_mean,
        action_std=dataset.action_std,
        device=args.device,
        max_seq_len=config['max_seq_len']
    )

    print(f"\n{'='*60}")
    print(f"Controller ready for Isaac Sim!")
    print(f"{'='*60}")
    print(f"Track: {track_name}")
    print(f"Gates: {len(gates)}")
    print(f"Control frequency: {args.control_freq} Hz")
    print(f"\nIntegration instructions:")
    print(f"1. In Isaac Sim, create drone and gate objects")
    print(f"2. In your Isaac Sim script, import this controller")
    print(f"3. Call controller.predict_action(current_state) at {args.control_freq}Hz")
    print(f"4. Apply returned action [thrust, roll, pitch, yaw] to drone")
    print(f"{'='*60}")

    # Example integration code
    print(f"\n📋 Example Isaac Sim integration code:")
    print("""
# In your Isaac Sim Python script:

from scripts.isaac_sim_inference import IsaacSimICLController, load_checkpoint, load_demo_episodes, load_track_gates
from drone_icl.dataset_icl import ICLRacingDataset

# Load model and controller (run once at startup)
model, config = load_checkpoint('path/to/checkpoint.pt', 'cuda')
dataset = ICLRacingDataset(h5_path='path/to/data.h5', ...)
demo_states, demo_actions, demo_masks = load_demo_episodes(['ep_000001', ...], dataset)
gates, track_name = load_track_gates('path/to/gates.json')

controller = IsaacSimICLController(
    model=model,
    demo_states=demo_states,
    demo_actions=demo_actions,
    demo_masks=demo_masks,
    state_mean=dataset.state_mean,
    state_std=dataset.state_std,
    action_mean=dataset.action_mean,
    action_std=dataset.action_std,
    device='cuda'
)

# Reset at episode start
controller.reset()
current_gate_idx = 0

# Control loop (called at 50Hz)
while not done:
    # Get drone state from Isaac Sim
    drone_pose = get_drone_pose()  # (7,) [pos, quat]
    drone_velocity = get_drone_velocity()  # (3,)
    drone_angular_velocity = get_drone_angular_velocity()  # (3,)

    # Construct state vector
    state = controller.get_state_from_isaac(
        drone_pose, drone_velocity, drone_angular_velocity,
        gates, current_gate_idx
    )

    # Predict action
    action = controller.predict_action(state)  # (4,) [thrust, roll, pitch, yaw]

    # Apply action to drone
    apply_drone_action(action)

    # Check gate passing
    if distance_to_gate(gates[current_gate_idx]) < gate_radius:
        current_gate_idx += 1
        print(f"Gate {current_gate_idx} passed!")

    # Check completion
    if current_gate_idx >= len(gates):
        print("All gates passed! Episode complete.")
        break
    """)

    print(f"\n📁 Track configuration format (JSON):")
    print("""
{
    "track_name": "ellipse",
    "gates": [
        {"id": 0, "position": [10.0, 0.0, 2.0]},
        {"id": 1, "position": [8.0, 5.0, 2.5]},
        {"id": 2, "position": [0.0, 8.0, 3.0]},
        ...
    ]
}
    """)

    # Return controller for use in Isaac Sim
    return controller, gates


if __name__ == '__main__':
    main()
