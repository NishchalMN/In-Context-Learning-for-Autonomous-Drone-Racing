"""
Training script for ICL Transformer Policy

Trains the model using next-token prediction on demonstration trajectories.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_icl.models import ICLTransformerPolicy
from drone_icl.dataset_icl import ICLRacingDataset, collate_racing_batch


class ICLTrainer:
    """Trainer for ICL Transformer Policy."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler (reduce on plateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.start_epoch = 1

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')

        for batch in pbar:
            # Move to device
            demo_states = batch['demo_states'].to(self.device)
            demo_actions = batch['demo_actions'].to(self.device)
            demo_masks = batch['demo_masks'].to(self.device)
            target_states = batch['target_states'].to(self.device)
            target_actions = batch['target_actions'].to(self.device)
            target_mask = batch['target_mask'].to(self.device)

            # Get images if available
            target_images = None
            if 'target_images' in batch:
                target_images = batch['target_images'].to(self.device)

            # Forward pass
            predicted_actions = self.model(
                demo_states,
                demo_actions,
                target_states,
                demo_mask=demo_masks,
                current_images=target_images
            )

            # Compute loss (only on valid timesteps)
            loss = self.compute_masked_loss(
                predicted_actions,
                target_actions,
                target_mask
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, epoch: int) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')

            for batch in pbar:
                # Move to device
                demo_states = batch['demo_states'].to(self.device)
                demo_actions = batch['demo_actions'].to(self.device)
                demo_masks = batch['demo_masks'].to(self.device)
                target_states = batch['target_states'].to(self.device)
                target_actions = batch['target_actions'].to(self.device)
                target_mask = batch['target_mask'].to(self.device)

                # Get images if available
                target_images = None
                if 'target_images' in batch:
                    target_images = batch['target_images'].to(self.device)

                # Forward pass
                predicted_actions = self.model(
                    demo_states,
                    demo_actions,
                    target_states,
                    demo_mask=demo_masks,
                    current_images=target_images
                )

                # Compute loss
                loss = self.compute_masked_loss(
                    predicted_actions,
                    target_actions,
                    target_mask
                )

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def compute_masked_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss only on valid (non-padded) timesteps.

        Args:
            predicted: (batch, seq_len, action_dim)
            target: (batch, seq_len, action_dim)
            mask: (batch, seq_len) - True for valid

        Returns:
            loss: scalar
        """
        # Expand mask to match action dimensions
        mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)

        # Compute squared error
        squared_error = (predicted - target) ** 2

        # Mask out invalid timesteps
        masked_error = squared_error * mask

        # Average over valid timesteps
        loss = masked_error.sum() / mask.sum()

        return loss

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        print(f"\n🔄 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"✅ Resumed from epoch {self.start_epoch - 1}")
        print(f"   Best val loss: {self.best_val_loss:.6f}")
        print(f"   Training history: {len(self.train_losses)} epochs")
        print(f"   Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")

    def train(self, num_epochs: int):
        """Full training loop."""
        print("=" * 70)
        print("  Starting Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        if self.start_epoch > 1:
            print(f"Resuming from epoch: {self.start_epoch}")
            print(f"Training until epoch: {num_epochs}")
        print("=" * 70)
        print()

        for epoch in range(self.start_epoch, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)

            # Step learning rate scheduler
            self.scheduler.step(val_loss)

            # Print summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")

            # Save checkpoint if best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f" New best model found -> Val loss: {val_loss:.6f}")

            # Save regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)

        print()
        print("=" * 70)
        print("  Training Complete!")
        print("=" * 70)
        print(f"Best validation loss: {self.best_val_loss:.6f}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'

        torch.save(checkpoint, path)

        # Also save training log
        log_data = {
            'epoch': epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

        with open(self.log_dir / 'training_log.json', 'w') as f:
            json.dump(log_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train ICL Transformer Policy')
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--val-data', type=str, default=None, help='Path to separate validation HDF5 dataset (optional)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-demos', type=int, default=2, help='Number of demo trajectories')
    parser.add_argument('--max-seq-len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split (ignored if --val-data is provided)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')

    # Vision arguments
    parser.add_argument('--use-vision', action='store_true', help='Enable vision encoder')
    parser.add_argument('--vision-backbone', type=str, default='simple_cnn',
                        choices=['simple_cnn', 'resnet18'], help='Vision backbone architecture')
    parser.add_argument('--use-optical-flow', action='store_true', help='Enable optical flow encoding')

    # Data augmentation arguments
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--augment-prob', type=float, default=0.5, help='Augmentation probability')
    parser.add_argument('--use-gate-info', action='store_true', default=False,
                       help='Add gate information to state (direction, distance, relative pos). Enables 26D state instead of 19D')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--save-freq', type=int, default=10, help='Checkpoint save frequency')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from (e.g., checkpoints/best_model.pt)')

    args = parser.parse_args()

    # Adjust batch size if using vision
    if args.use_vision and args.batch_size > 8:
        print(f"⚠️  Warning: Batch size {args.batch_size} may be too large with vision.")
        print(f"   Recommended: 4-8 for simple_cnn, 2-4 for resnet18")
        print(f"   Consider reducing --batch-size if you encounter OOM errors.")

    # Create dataset
    print("Loading dataset...")
    print(f"Vision: {'Enabled' if args.use_vision else 'Disabled'}")
    if args.use_vision:
        print(f"  Backbone: {args.vision_backbone}")
        print(f"  Optical flow: {'Enabled' if args.use_optical_flow else 'Disabled'}")

    print(f"Data augmentation: {'Enabled' if args.augment else 'Disabled'}")
    if args.augment:
        print(f"  Augmentation probability: {args.augment_prob}")

    # Check if separate validation dataset is provided
    if args.val_data is not None:
        print("\n✅ Using separate train and validation datasets (NO DATA LEAKAGE)")
        print(f"   Train data: {args.data}")
        print(f"   Val data:   {args.val_data}")

        # Load training dataset
        train_dataset = ICLRacingDataset(
            h5_path=args.data,
            num_demos=args.num_demos,
            max_seq_len=args.max_seq_len,
            use_images=args.use_vision,
            normalize=True,
            subsample_factor=10,
            augment=args.augment,
            augment_prob=args.augment_prob,
            use_gate_info=args.use_gate_info
        )

        # Load validation dataset (NO augmentation)
        val_dataset = ICLRacingDataset(
            h5_path=args.val_data,
            num_demos=args.num_demos,
            max_seq_len=args.max_seq_len,
            use_images=args.use_vision,
            normalize=True,
            subsample_factor=10,
            augment=False,  # Never augment validation
            augment_prob=0.0,
            use_gate_info=args.use_gate_info
        )

        train_size = len(train_dataset)
        val_size = len(val_dataset)

        print(f"\nTrain windows: {train_size}")
        print(f"Val windows:   {val_size}")

    else:
        # Original behavior: split single dataset
        print("\n⚠️  Using single dataset with train/val split")
        print(f"   Dataset: {args.data}")

        full_dataset = ICLRacingDataset(
            h5_path=args.data,
            num_demos=args.num_demos,
            max_seq_len=args.max_seq_len,
            use_images=args.use_vision,
            normalize=True,
            subsample_factor=10,
            augment=args.augment,
            augment_prob=args.augment_prob,
            use_gate_info=args.use_gate_info
        )

        # Train/val split - EPISODE-LEVEL to prevent data leakage
        print("\nSplitting dataset by original episodes (no window overlap)...")

        # Group windows by original episode
        original_episodes = {}
        for ep_name in full_dataset.episodes:
            # Extract base episode name (remove _wXXX suffix from sliding windows)
            # e.g., "ep_000000_w005" -> "ep_000000"
            base_name = '_'.join(ep_name.split('_')[:-1])
            if base_name not in original_episodes:
                original_episodes[base_name] = []
            original_episodes[base_name].append(ep_name)

        print(f"Found {len(original_episodes)} original episodes")

        # Split original episodes (not windows!)
        base_eps = sorted(list(original_episodes.keys()))
        np.random.seed(42)
        np.random.shuffle(base_eps)

        split_idx = int(len(base_eps) * (1 - args.val_split))
        train_base_eps = base_eps[:split_idx]
        val_base_eps = base_eps[split_idx:]

        print(f"Train original episodes: {len(train_base_eps)}")
        print(f"Val original episodes: {len(val_base_eps)}")

        # Create episode lists (all windows from each base episode)
        train_ep_names = []
        for base in train_base_eps:
            train_ep_names.extend(original_episodes[base])

        val_ep_names = []
        for base in val_base_eps:
            val_ep_names.extend(original_episodes[base])

        # Convert to indices
        train_indices = [full_dataset.episodes.index(ep) for ep in train_ep_names]
        val_indices = [full_dataset.episodes.index(ep) for ep in val_ep_names]

        # Create subsets
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        train_size = len(train_dataset)
        val_size = len(val_dataset)

        print(f"Train windows: {train_size}, Val windows: {val_size}")
        print(f"✅ No data leakage - all windows from same episode stay together")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_racing_batch,
        num_workers=args.num_workers  # Use specified num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_racing_batch,
        num_workers=args.num_workers
    )

    # Determine state dimension based on gate_info flag
    state_dim = 26 if args.use_gate_info else 19
    print(f"Gate information: {'Enabled' if args.use_gate_info else 'Disabled'}")
    if args.use_gate_info:
        print(f"  State dim: {state_dim} (19 base + 7 gate features)")
        print(f"  Gate features: direction (3D), distance (1D), relative pos (3D)")

    # Create model
    print("Creating model...")
    model = ICLTransformerPolicy(
        state_dim=state_dim,
        action_dim=4,
        embed_dim=args.embed_dim,
        num_heads=8,
        num_decoder_layers=6,
        dropout=0.1,
        use_vision=args.use_vision,
        vision_backbone=args.vision_backbone,
        use_optical_flow=args.use_optical_flow
    )

    # Create trainer
    trainer = ICLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
