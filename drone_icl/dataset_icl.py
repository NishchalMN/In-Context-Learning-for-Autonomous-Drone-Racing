"""
PyTorch Dataset for ICL Racing Data with Track Grouping

This dataset properly groups episodes by track type for in-context learning.
Multiple demonstrations of the same track are used as context.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict


class ICLRacingDataset(Dataset):
    """
    Dataset for drone racing with In-Context Learning.

    Key difference from basic dataset:
    - Groups episodes by track_type (same racing course)
    - Samples demos from the SAME track as target
    - Essential for ICL to work properly
    - Adds gate information (relative position, distance, direction) to state
    """

    def __init__(
        self,
        h5_path: str,
        num_demos: int = 2,
        max_seq_len: Optional[int] = 512,
        use_images: bool = False,
        normalize: bool = True,
        subsample_factor: int = 1,
        augment: bool = False,
        augment_prob: float = 0.5,
        use_gate_info: bool = False
    ):
        """
        Args:
            h5_path: Path to HDF5 file
            num_demos: Number of demonstration trajectories for context
            max_seq_len: Maximum sequence length (truncate if longer)
            use_images: Whether to include RGB images
            normalize: Whether to normalize states/actions
            subsample_factor: Downsample trajectories by this factor (e.g., 10 for 500Hz->50Hz)
            augment: Whether to apply data augmentation
            augment_prob: Probability of applying each augmentation (default 0.5)
            use_gate_info: Whether to add gate information to state (default False)
        """
        self.h5_path = Path(h5_path)
        self.num_demos = num_demos
        self.max_seq_len = max_seq_len
        self.use_images = use_images
        self.normalize = normalize
        self.subsample_factor = subsample_factor
        self.augment = augment
        self.augment_prob = augment_prob
        self.use_gate_info = use_gate_info

        # Open HDF5 file
        self.h5_file = h5py.File(h5_path, 'r')
        self.episodes = sorted(list(self.h5_file['episodes'].keys()))

        print(f"Loaded {len(self.episodes)} episodes from {h5_path}")

        # Group episodes by track type for ICL
        self.track_to_episodes = defaultdict(list)
        self.episode_to_track = {}

        for ep_name in self.episodes:
            ep = self.h5_file[f'episodes/{ep_name}']
            track_type = ep.attrs.get('track_type', 'unknown')
            self.track_to_episodes[track_type].append(ep_name)
            self.episode_to_track[ep_name] = track_type

        print(f"\nTrack grouping for ICL:")
        for track_type, episodes in sorted(self.track_to_episodes.items()):
            print(f"  {track_type}: {len(episodes)} episodes")

        # Compute normalization statistics
        if self.normalize:
            self._compute_normalization_stats()

    @staticmethod
    def _compute_gate_features(positions, gates):
        """
        Compute gate-relative features for each timestep.

        Args:
            positions: (T, 3) numpy array of drone positions
            gates: (N, 3) numpy array of gate positions

        Returns:
            gate_features: (T, 7) numpy array with:
                - next_gate_direction (3D): normalized direction to next gate
                - next_gate_distance (1D): distance to next gate
                - next_gate_relative_pos (3D): relative position to next gate
        """
        T = len(positions)
        N = len(gates)
        gate_features = np.zeros((T, 7), dtype=np.float32)

        for t in range(T):
            pos = positions[t]

            # Find closest gate ahead (simple heuristic: closest gate)
            distances = np.linalg.norm(gates - pos, axis=1)
            next_gate_idx = np.argmin(distances)
            next_gate = gates[next_gate_idx]

            # Relative position to next gate
            relative_pos = next_gate - pos  # (3,)
            distance = np.linalg.norm(relative_pos)

            # Normalized direction (handle zero distance)
            if distance > 1e-6:
                direction = relative_pos / distance
            else:
                direction = np.array([1.0, 0.0, 0.0])  # Default forward

            # Pack features: [direction (3), distance (1), relative_pos (3)]
            gate_features[t, 0:3] = direction
            gate_features[t, 3] = distance
            gate_features[t, 4:7] = relative_pos

        return gate_features

    def _compute_normalization_stats(self):
        """Compute mean/std for states and actions across all episodes."""
        print("\nComputing normalization statistics...")

        all_states = []
        all_actions = []

        # Sample subset for faster computation (use every 10th episode)
        sample_episodes = self.episodes[::10]
        print(f"  Sampling {len(sample_episodes)} / {len(self.episodes)} episodes for stats")

        for ep_name in sample_episodes:
            ep = self.h5_file[f'episodes/{ep_name}']

            # Subsample if needed
            T = ep['actions'].shape[0]
            indices = np.arange(0, T, self.subsample_factor)

            # Concatenate state components
            state = np.concatenate([
                ep['state/pos'][indices],
                ep['state/quat'][indices],
                ep['state/lin_vel'][indices],
                ep['state/ang_vel'][indices],
                ep['imu/lin_acc'][indices],
                ep['imu/ang_vel'][indices]
            ], axis=-1)

            # Add gate features if enabled
            if self.use_gate_info:
                positions = ep['state/pos'][indices]
                gates = ep['track_gates'][:]
                gate_features = self._compute_gate_features(positions, gates)
                state = np.concatenate([state, gate_features], axis=-1)

            actions = ep['actions'][indices]

            all_states.append(state)
            all_actions.append(actions)

        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)

        self.state_mean = torch.FloatTensor(all_states.mean(axis=0))
        self.state_std = torch.FloatTensor(all_states.std(axis=0) + 1e-8)
        self.action_mean = torch.FloatTensor(all_actions.mean(axis=0))
        self.action_std = torch.FloatTensor(all_actions.std(axis=0) + 1e-8)

        print(f"  State dim: {self.state_mean.shape[0]} {'(with gate info)' if self.use_gate_info else ''}")
        print(f"  Action dim: {self.action_mean.shape[0]}")
        print(f"  State range: [{all_states.min():.2f}, {all_states.max():.2f}]")
        print(f"  Action range: [{all_actions.min():.2f}, {all_actions.max():.2f}]")

    def __len__(self) -> int:
        return len(self.episodes)

    def _augment_trajectory_fixed_length(self, state: torch.Tensor, actions: torch.Tensor,
                                         gates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation that doesn't change sequence length.

        Augmentations (applied with probability augment_prob):
        1. Time reversal: Flip trajectory backwards
        2. Horizontal flip: Mirror Y-axis (for symmetric tracks)
        3. Gaussian noise: Add small noise to states

        NOTE: Speed variation removed to avoid length mismatches in batch collation

        Args:
            state: (T, 19) or (T, 26) state tensor (with or without gate info)
            actions: (T, 4) action tensor
            gates: (N, 3) gate positions

        Returns:
            Augmented state, actions, gates (same length as input)
        """
        state_dim = state.shape[-1]
        has_gate_info = (state_dim == 26)  # 19 base + 7 gate features

        # 1. Time reversal (50% chance)
        if np.random.rand() < self.augment_prob:
            state = torch.flip(state, dims=[0])
            actions = torch.flip(actions, dims=[0])

        # 2. Horizontal flip (30% chance)
        # Flip Y-axis: pos_y, vel_y, quat (affects orientation)
        if np.random.rand() < self.augment_prob * 0.6:
            # Position Y (index 1)
            state[:, 1] *= -1
            # Lin vel Y (index 7 + 1 = 8, after pos(3) + quat(4))
            state[:, 8] *= -1
            # Quaternion Y (index 4) - simplified, not fully correct but ok for augmentation
            state[:, 4] *= -1
            # Action: roll (index 1)
            actions[:, 1] *= -1
            # Gates Y
            gates[:, 1] *= -1

            # If gate info present, flip gate direction Y and relative pos Y
            if has_gate_info:
                # Gate direction Y (index 19 + 1 = 20)
                state[:, 20] *= -1
                # Gate relative pos Y (index 19 + 4 + 1 = 24)
                state[:, 24] *= -1

        # 3. Gaussian noise (50% chance)
        if np.random.rand() < self.augment_prob:
            # Add small noise to state (1% of std)
            noise_scale = 0.01
            state_noise = torch.randn_like(state) * noise_scale
            state = state + state_noise

            # Add small noise to actions (1% of std)
            action_noise = torch.randn_like(actions) * noise_scale
            actions = actions + action_noise

        return state, actions, gates

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a training sample with demos from the SAME track.

        This is critical for ICL: the model learns to adapt to a new
        trajectory on the same track by conditioning on demo trajectories.
        """
        # Select target episode
        target_ep_name = self.episodes[idx]
        target_track = self.episode_to_track[target_ep_name]

        # Get all episodes from the SAME track
        same_track_episodes = self.track_to_episodes[target_track]

        # Select demo episodes (excluding target)
        available_demos = [ep for ep in same_track_episodes if ep != target_ep_name]

        if len(available_demos) >= self.num_demos:
            # Randomly sample demos
            demo_ep_names = np.random.choice(
                available_demos,
                size=self.num_demos,
                replace=False
            ).tolist()
        else:
            # If not enough demos, use all available and pad with repeats
            demo_ep_names = available_demos.copy()
            while len(demo_ep_names) < self.num_demos:
                demo_ep_names.append(np.random.choice(available_demos))

        # Load target trajectory
        target_data = self._load_episode(target_ep_name)

        # Load demo trajectories
        demo_data = [self._load_episode(name) for name in demo_ep_names]

        # Stack demos
        demo_states = torch.stack([d['states'] for d in demo_data])
        demo_actions = torch.stack([d['actions'] for d in demo_data])
        demo_masks = torch.stack([d['mask'] for d in demo_data])

        result = {
            'demo_states': demo_states,
            'demo_actions': demo_actions,
            'demo_masks': demo_masks,
            'target_states': target_data['states'],
            'target_actions': target_data['actions'],
            'target_mask': target_data['mask'],
            'gates': target_data['gates'],
            'gates_passed': target_data['gates_passed'],
            'episode_id': idx,
            'track_type': target_track
        }

        # Add images if available
        if self.use_images and 'images' in demo_data[0]:
            demo_images = torch.stack([d['images'] for d in demo_data])
            result['demo_images'] = demo_images
            result['target_images'] = target_data['images']

        return result

    def _load_episode(self, ep_name: str) -> Dict[str, torch.Tensor]:
        """Load a single episode and convert to tensors."""
        ep = self.h5_file[f'episodes/{ep_name}']

        # Get trajectory length
        T = ep['actions'].shape[0]

        # Subsample if needed
        indices = np.arange(0, T, self.subsample_factor)

        # Concatenate state components
        state = np.concatenate([
            ep['state/pos'][indices],
            ep['state/quat'][indices],
            ep['state/lin_vel'][indices],
            ep['state/ang_vel'][indices],
            ep['imu/lin_acc'][indices],
            ep['imu/ang_vel'][indices]
        ], axis=-1)

        actions = ep['actions'][indices]
        gates = ep['track_gates'][:]

        # Add gate features if enabled
        if self.use_gate_info:
            positions = ep['state/pos'][indices]
            gate_features = self._compute_gate_features(positions, gates)
            state = np.concatenate([state, gate_features], axis=-1)

        # gates_passed might not exist in expanded dataset
        if 'gates_passed' in ep:
            gates_passed = ep['gates_passed'][indices]
        else:
            # Create dummy gates_passed (not used in training)
            gates_passed = np.zeros(len(indices), dtype=np.int64)

        # Load RGB images if requested
        images = None
        if self.use_images and 'rgb' in ep:
            images = ep['rgb'][indices]

        # Convert to tensors
        state = torch.FloatTensor(state)
        actions = torch.FloatTensor(actions)
        gates = torch.FloatTensor(gates)
        gates_passed = torch.LongTensor(gates_passed)

        # Convert images to tensor
        if images is not None:
            # (T, H, W, 3) uint8 -> (T, 3, H, W) float32
            images = torch.FloatTensor(images).permute(0, 3, 1, 2) / 255.0

        # Normalize states/actions
        if self.normalize:
            state = (state - self.state_mean) / self.state_std
            actions = (actions - self.action_mean) / self.action_std

        # Get actual length after subsampling
        T_sub = len(state)

        # Apply data augmentation ONLY AFTER getting length and BEFORE padding
        # This ensures augmentation doesn't affect padding logic
        if self.augment:
            # Only apply non-length-changing augmentations
            # Skip speed variation to avoid length mismatches in collate_fn
            state, actions, gates = self._augment_trajectory_fixed_length(state, actions, gates)

        # Truncate or pad to max_seq_len
        if self.max_seq_len is not None:
            if T_sub > self.max_seq_len:
                # Truncate
                state = state[:self.max_seq_len]
                actions = actions[:self.max_seq_len]
                gates_passed = gates_passed[:self.max_seq_len]
                if images is not None:
                    images = images[:self.max_seq_len]
                mask = torch.ones(self.max_seq_len, dtype=torch.bool)
            else:
                # Pad
                pad_len = self.max_seq_len - T_sub
                state = torch.cat([
                    state,
                    torch.zeros(pad_len, state.shape[-1])
                ], dim=0)
                actions = torch.cat([
                    actions,
                    torch.zeros(pad_len, actions.shape[-1])
                ], dim=0)
                gates_passed = torch.cat([
                    gates_passed,
                    torch.zeros(pad_len, dtype=torch.long)
                ], dim=0)
                if images is not None:
                    _, C, H, W = images.shape
                    images = torch.cat([
                        images,
                        torch.zeros(pad_len, C, H, W)
                    ], dim=0)
                mask = torch.cat([
                    torch.ones(T_sub, dtype=torch.bool),
                    torch.zeros(pad_len, dtype=torch.bool)
                ], dim=0)
        else:
            mask = torch.ones(T_sub, dtype=torch.bool)

        result = {
            'states': state,
            'actions': actions,
            'gates': gates,
            'gates_passed': gates_passed,
            'mask': mask
        }

        if images is not None:
            result['images'] = images

        return result

    def close(self):
        """Close HDF5 file."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def __del__(self):
        self.close()


def collate_racing_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    result = {
        'demo_states': torch.stack([b['demo_states'] for b in batch]),
        'demo_actions': torch.stack([b['demo_actions'] for b in batch]),
        'demo_masks': torch.stack([b['demo_masks'] for b in batch]),
        'target_states': torch.stack([b['target_states'] for b in batch]),
        'target_actions': torch.stack([b['target_actions'] for b in batch]),
        'target_mask': torch.stack([b['target_mask'] for b in batch]),
        'gates': torch.stack([b['gates'] for b in batch]),
        'gates_passed': torch.stack([b['gates_passed'] for b in batch]),
        'episode_id': torch.LongTensor([b['episode_id'] for b in batch])
    }

    # Add images if present
    if 'demo_images' in batch[0]:
        result['demo_images'] = torch.stack([b['demo_images'] for b in batch])
        result['target_images'] = torch.stack([b['target_images'] for b in batch])

    return result


if __name__ == "__main__":
    # Test the dataset
    dataset = ICLRacingDataset(
        "data/ratm_racing_dataset.h5",
        num_demos=2,
        max_seq_len=256,
        subsample_factor=10  # 500Hz -> 50Hz
    )

    print(f"\nDataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]

    print("\nSample structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {value}")

    # Test dataloader
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_racing_batch
    )

    batch = next(iter(loader))
    print("\nBatch structure:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")

    dataset.close()
