"""
Encoders for ICL Drone Racing

Components:
- ObservationEncoder: Encodes current state to embedding
- TrajectoryEncoder: Encodes demonstration trajectories to context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ObservationEncoder(nn.Module):
    """
    Encodes observation (state vector) to embedding.

    Input: (batch, state_dim) where state_dim = 19
           [pos(3), quat(4), lin_vel(3), ang_vel(3), imu_acc(3), imu_gyro(3)]
    Output: (batch, embed_dim)
    """

    def __init__(
        self,
        state_dim: int = 19,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # MLP encoder
        layers = []
        in_dim = state_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Final projection to embed_dim
        layers.append(nn.Linear(in_dim, embed_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim) or (batch, seq_len, state_dim)

        Returns:
            embedding: (batch, embed_dim) or (batch, seq_len, embed_dim)
        """
        return self.encoder(state)


class TrajectoryEncoder(nn.Module):
    """
    Encodes demonstration trajectory (state-action pairs) to context embeddings.

    Uses a transformer encoder to capture temporal structure.

    Input:
        - states: (batch, num_demos, seq_len, state_dim)
        - actions: (batch, num_demos, seq_len, action_dim)
        - mask: (batch, num_demos, seq_len) - True for valid timesteps

    Output:
        - context: (batch, num_demos * seq_len, embed_dim)
    """

    def __init__(
        self,
        state_dim: int = 19,
        action_dim: int = 4,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # Embed state-action pairs
        self.state_action_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, dropout, max_len=1024)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            states: (batch, num_demos, seq_len, state_dim)
            actions: (batch, num_demos, seq_len, action_dim)
            mask: (batch, num_demos, seq_len) - True for valid

        Returns:
            context: (batch, num_demos * seq_len, embed_dim)
        """
        batch_size, num_demos, seq_len, _ = states.shape

        # Concatenate state and action
        state_action = torch.cat([states, actions], dim=-1)  # (B, D, T, state+action)

        # Reshape to process all demos together
        state_action = state_action.view(batch_size * num_demos, seq_len, -1)

        # Encode state-action pairs
        embeddings = self.state_action_encoder(state_action)  # (B*D, T, embed_dim)

        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)

        # Create padding mask for transformer (simpler approach)
        if mask is not None:
            # Reshape mask: (B, D, T) -> (B*D, T)
            key_padding_mask = mask.view(batch_size * num_demos, seq_len)
            # Transformer expects: True for positions to IGNORE
            key_padding_mask = ~key_padding_mask  # Invert
        else:
            key_padding_mask = None

        # Apply transformer
        context = self.transformer(embeddings, src_key_padding_mask=key_padding_mask)  # (B*D, T, embed_dim)

        # Reshape back to separate demos
        # context shape after transformer: (B*D, T, embed_dim)
        actual_seq_len = context.shape[1]  # Use actual length from transformer output
        context = context.view(batch_size, num_demos * actual_seq_len, self.embed_dim)

        return context


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            (batch, seq_len, embed_dim)
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)


class VisionEncoder(nn.Module):
    """
    Encodes RGB images to visual embeddings.

    Options:
    - CNN backbone (ResNet-18)
    - Optionally compute optical flow between frames

    Input: (batch, seq_len, 3, H, W) - RGB images
    Output: (batch, seq_len, embed_dim) - visual features
    """

    def __init__(
        self,
        embed_dim: int = 256,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        use_optical_flow: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_optical_flow = use_optical_flow

        if backbone == 'resnet18':
            import torchvision.models as models
            # Load pretrained ResNet18
            resnet = models.resnet18(pretrained=pretrained)

            # Remove final FC layer
            # ResNet18 outputs 512-dim features
            self.cnn = nn.Sequential(*list(resnet.children())[:-1])
            cnn_output_dim = 512

        elif backbone == 'simple_cnn':
            # Lightweight CNN for faster training
            self.cnn = nn.Sequential(
                # Input: (B*T, 3, 480, 640)
                nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),  # -> (B*T, 32, 120, 160)
                nn.ReLU(),
                nn.MaxPool2d(2),  # -> (B*T, 32, 60, 80)

                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # -> (B*T, 64, 30, 40)
                nn.ReLU(),
                nn.MaxPool2d(2),  # -> (B*T, 64, 15, 20)

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B*T, 128, 8, 10)
                nn.ReLU(),
                nn.MaxPool2d(2),  # -> (B*T, 128, 4, 5)

                nn.AdaptiveAvgPool2d((1, 1))  # -> (B*T, 128, 1, 1)
            )
            cnn_output_dim = 128
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Project CNN features to embed_dim
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_output_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Optical flow encoder (if enabled)
        if use_optical_flow:
            self.flow_encoder = nn.Sequential(
                # Flow is 2-channel (x, y velocities)
                nn.Conv2d(2, 32, kernel_size=7, stride=4, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, embed_dim // 2)
            )

            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim + embed_dim // 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )

    def forward(
        self,
        images: torch.Tensor,
        optical_flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            images: (batch, seq_len, 3, H, W) - RGB images
            optical_flow: (batch, seq_len, 2, H, W) - Optional optical flow

        Returns:
            visual_features: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, C, H, W = images.shape

        # Reshape to process all frames together
        images_flat = images.view(batch_size * seq_len, C, H, W)

        # Extract CNN features
        cnn_features = self.cnn(images_flat)  # (B*T, cnn_dim, 1, 1) or (B*T, cnn_dim)

        # Project to embed_dim
        visual_features = self.projection(cnn_features)  # (B*T, embed_dim)

        # Reshape back
        visual_features = visual_features.view(batch_size, seq_len, self.embed_dim)

        # Add optical flow if available
        if self.use_optical_flow and optical_flow is not None:
            flow_flat = optical_flow.view(batch_size * seq_len, 2, H, W)
            flow_features = self.flow_encoder(flow_flat)  # (B*T, embed_dim//2)
            flow_features = flow_features.view(batch_size, seq_len, -1)

            # Concatenate and fuse
            combined = torch.cat([visual_features, flow_features], dim=-1)
            visual_features = self.fusion(combined)

        return visual_features


class MultiModalEncoder(nn.Module):
    """
    Fuses state and visual information.

    Input:
        - states: (batch, seq_len, state_dim)
        - images: (batch, seq_len, 3, H, W)

    Output:
        - fused_features: (batch, seq_len, embed_dim)
    """

    def __init__(
        self,
        state_dim: int = 19,
        embed_dim: int = 256,
        vision_backbone: str = 'simple_cnn',
        use_optical_flow: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # State encoder
        self.state_encoder = ObservationEncoder(
            state_dim=state_dim,
            embed_dim=embed_dim,
            dropout=dropout
        )

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            embed_dim=embed_dim,
            backbone=vision_backbone,
            use_optical_flow=use_optical_flow,
            dropout=dropout
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        states: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        optical_flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            states: (batch, seq_len, state_dim)
            images: (batch, seq_len, 3, H, W) - optional
            optical_flow: (batch, seq_len, 2, H, W) - optional

        Returns:
            fused: (batch, seq_len, embed_dim)
        """
        # Encode state
        state_features = self.state_encoder(states)  # (batch, seq_len, embed_dim)

        # If no images, just return state features
        if images is None:
            return state_features

        # Encode vision
        visual_features = self.vision_encoder(images, optical_flow)  # (batch, seq_len, embed_dim)

        # Concatenate and fuse
        combined = torch.cat([state_features, visual_features], dim=-1)
        fused = self.fusion(combined)

        return fused


if __name__ == "__main__":
    # Test encoders
    print("=" * 70)
    print("  Testing Encoders")
    print("=" * 70)
    print()

    batch_size = 4
    state_dim = 19
    action_dim = 4
    embed_dim = 256

    # Test ObservationEncoder
    print("1. Testing ObservationEncoder...")
    obs_encoder = ObservationEncoder(state_dim=state_dim, embed_dim=embed_dim)

    # Single observation
    state = torch.randn(batch_size, state_dim)
    obs_embed = obs_encoder(state)
    print(f"   Input: {state.shape}")
    print(f"   Output: {obs_embed.shape}")
    assert obs_embed.shape == (batch_size, embed_dim), "Shape mismatch!"
    print("   ✅ ObservationEncoder works!")
    print()

    # Test TrajectoryEncoder
    print("2. Testing TrajectoryEncoder...")
    traj_encoder = TrajectoryEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim
    )

    num_demos = 2
    seq_len = 128

    demo_states = torch.randn(batch_size, num_demos, seq_len, state_dim)
    demo_actions = torch.randn(batch_size, num_demos, seq_len, action_dim)
    demo_mask = torch.ones(batch_size, num_demos, seq_len, dtype=torch.bool)

    context = traj_encoder(demo_states, demo_actions, demo_mask)
    print(f"   Input states: {demo_states.shape}")
    print(f"   Input actions: {demo_actions.shape}")
    print(f"   Output context: {context.shape}")
    assert context.shape == (batch_size, num_demos * seq_len, embed_dim), "Shape mismatch!"
    print("   ✅ TrajectoryEncoder works!")
    print()

    print("=" * 70)
    print("✅ All encoders working correctly!")
    print("=" * 70)
