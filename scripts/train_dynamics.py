"""
Train a forward dynamics model for drone racing.

Learns state transitions: (state_t, action_t) -> state_t+1
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_icl.dataset_icl import ICLRacingDataset


class SimpleDynamicsModel(nn.Module):
    """
    Simple learned forward dynamics: state_t, action_t -> state_t+1
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


def train_dynamics_model(dataset, epochs=50, batch_size=128, device='cpu',
                         checkpoint_dir=None):
    """
    Train a simple forward dynamics model on the dataset.

    Args:
        dataset: ICLRacingDataset
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
        checkpoint_dir: Where to save checkpoints

    Returns:
        Trained dynamics model
    """
    print("="*60)
    print("Training Dynamics Model")
    print("="*60)

    model = SimpleDynamicsModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Collect all state-action-next_state pairs
    print("\nCollecting training data...")
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
    print(f"\nTraining for {epochs} epochs...")
    model.train()
    best_loss = float('inf')

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

        if avg_loss < best_loss:
            best_loss = avg_loss
            if checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir) / 'best_model.pt'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, checkpoint_path)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")

    print(f"\n✅ Dynamics model trained! Best loss: {best_loss:.6f}")
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Train dynamics model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset HDF5')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/dynamics',
                       help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    parser.add_argument('--max-seq-len', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--subsample-factor', type=int, default=10,
                       help='Subsample factor for dataset')

    args = parser.parse_args()

    print(f"Loading dataset: {args.data}")
    dataset = ICLRacingDataset(
        h5_path=args.data,
        num_demos=1,  # Not used for dynamics training
        max_seq_len=args.max_seq_len,
        use_images=False,
        normalize=True,
        subsample_factor=args.subsample_factor
    )

    # Train model
    model = train_dynamics_model(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )

    # Save final model
    checkpoint_path = Path(args.checkpoint_dir) / 'final_model.pt'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    print(f"\nSaved final model to: {checkpoint_path}")


if __name__ == '__main__':
    main()
