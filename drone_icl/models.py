"""
ICL Transformer Policy for Drone Racing

Main model that performs in-context learning from demonstration trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from drone_icl.encoders import ObservationEncoder, TrajectoryEncoder, MultiModalEncoder


class ICLTransformerPolicy(nn.Module):
    """
    In-Context Learning Transformer Policy.

    Given:
    - Demonstration trajectories (state-action pairs from 2-3 laps)
    - Current state

    Predicts:
    - Next action to take

    Architecture:
    1. Encode demos → Context embeddings
    2. Encode current state → Query embedding
    3. Cross-attention: Query attends to Context
    4. Decode to action
    """

    def __init__(
        self,
        state_dim: int = 19,  # Can be 19 (base) or 26 (with gate info)
        action_dim: int = 4,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        max_demo_len: int = 512,
        use_vision: bool = False,
        vision_backbone: str = 'simple_cnn',
        use_optical_flow: bool = False
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.use_vision = use_vision

        # Observation encoder (state-only or multi-modal)
        if use_vision:
            self.obs_encoder = MultiModalEncoder(
                state_dim=state_dim,
                embed_dim=embed_dim,
                vision_backbone=vision_backbone,
                use_optical_flow=use_optical_flow,
                dropout=dropout
            )
        else:
            self.obs_encoder = ObservationEncoder(
                state_dim=state_dim,
                embed_dim=embed_dim,
                dropout=dropout
            )

        self.traj_encoder = TrajectoryEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Cross-attention decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, action_dim)
        )

    def forward(
        self,
        demo_states: torch.Tensor,
        demo_actions: torch.Tensor,
        current_states: torch.Tensor,
        demo_mask: Optional[torch.Tensor] = None,
        current_images: Optional[torch.Tensor] = None,
        current_optical_flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            demo_states: (batch, num_demos, demo_len, state_dim)
            demo_actions: (batch, num_demos, demo_len, action_dim)
            current_states: (batch, seq_len, state_dim) - current trajectory
            demo_mask: (batch, num_demos, demo_len) - valid timesteps
            current_images: (batch, seq_len, 3, H, W) - optional RGB images
            current_optical_flow: (batch, seq_len, 2, H, W) - optional optical flow

        Returns:
            predicted_actions: (batch, seq_len, action_dim)
        """
        batch_size = current_states.shape[0]
        seq_len = current_states.shape[1]

        # Encode demonstration trajectories → Context
        context = self.traj_encoder(demo_states, demo_actions, demo_mask)
        # context: (batch, num_demos * demo_len, embed_dim)

        # Encode current states → Query
        if self.use_vision and current_images is not None:
            query = self.obs_encoder(current_states, current_images, current_optical_flow)
        else:
            query = self.obs_encoder(current_states)
        # query: (batch, seq_len, embed_dim)

        # Cross-attention: Query attends to Context
        attended = self.decoder(query, context)
        # attended: (batch, seq_len, embed_dim)

        # Predict actions
        actions = self.action_head(attended)
        # actions: (batch, seq_len, action_dim)

        return actions

    def predict_single_step(
        self,
        demo_states: torch.Tensor,
        demo_actions: torch.Tensor,
        current_state: torch.Tensor,
        demo_mask: Optional[torch.Tensor] = None,
        current_image: Optional[torch.Tensor] = None,
        current_flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict action for a single state (inference mode).

        Args:
            demo_states: (batch, num_demos, demo_len, state_dim)
            demo_actions: (batch, num_demos, demo_len, action_dim)
            current_state: (batch, state_dim)
            demo_mask: (batch, num_demos, demo_len)
            current_image: (batch, 3, H, W) - optional
            current_flow: (batch, 2, H, W) - optional

        Returns:
            action: (batch, action_dim)
        """
        # Add sequence dimension
        current_state = current_state.unsqueeze(1)  # (batch, 1, state_dim)

        if current_image is not None:
            current_image = current_image.unsqueeze(1)  # (batch, 1, 3, H, W)

        if current_flow is not None:
            current_flow = current_flow.unsqueeze(1)  # (batch, 1, 2, H, W)

        # Forward pass
        actions = self.forward(
            demo_states, demo_actions, current_state, demo_mask,
            current_images=current_image,
            current_optical_flow=current_flow
        )

        # Remove sequence dimension
        action = actions.squeeze(1)  # (batch, action_dim)

        return action


class ICLTransformerPolicyAutoregressive(ICLTransformerPolicy):
    """
    Autoregressive version of ICL policy.

    Predicts actions autoregressively given demonstration context.
    This version is closer to GPT-style next-token prediction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add action embedding for autoregressive conditioning
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.embed_dim),
            nn.ReLU()
        )

    def forward(
        self,
        demo_states: torch.Tensor,
        demo_actions: torch.Tensor,
        current_states: torch.Tensor,
        current_actions: Optional[torch.Tensor] = None,
        demo_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            demo_states: (batch, num_demos, demo_len, state_dim)
            demo_actions: (batch, num_demos, demo_len, action_dim)
            current_states: (batch, seq_len, state_dim)
            current_actions: (batch, seq_len-1, action_dim) - previous actions
            demo_mask: (batch, num_demos, demo_len)

        Returns:
            predicted_actions: (batch, seq_len, action_dim)
        """
        batch_size, seq_len, _ = current_states.shape

        # Encode demonstrations → Context
        context = self.traj_encoder(demo_states, demo_actions, demo_mask)

        # Encode current states
        query = self.obs_encoder(current_states)

        # If previous actions provided, add them as conditioning
        if current_actions is not None:
            # Pad with zeros for first timestep (no previous action)
            zero_action = torch.zeros(batch_size, 1, self.action_dim, device=current_actions.device)
            padded_actions = torch.cat([zero_action, current_actions], dim=1)

            # Encode previous actions
            action_embed = self.action_encoder(padded_actions)

            # Add to query
            query = query + action_embed

        # Cross-attention
        attended = self.decoder(query, context)

        # Predict next actions
        actions = self.action_head(attended)

        return actions


if __name__ == "__main__":
    # Test the models
    print("=" * 70)
    print("  Testing ICL Transformer Policy")
    print("=" * 70)
    print()

    batch_size = 4
    num_demos = 2
    demo_len = 128
    seq_len = 64
    state_dim = 19
    action_dim = 4
    embed_dim = 256

    # Test standard ICL policy
    print("1. Testing ICLTransformerPolicy...")
    model = ICLTransformerPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim
    )

    demo_states = torch.randn(batch_size, num_demos, demo_len, state_dim)
    demo_actions = torch.randn(batch_size, num_demos, demo_len, action_dim)
    current_states = torch.randn(batch_size, seq_len, state_dim)
    demo_mask = torch.ones(batch_size, num_demos, demo_len, dtype=torch.bool)

    predicted_actions = model(demo_states, demo_actions, current_states, demo_mask)

    print(f"   Demo states: {demo_states.shape}")
    print(f"   Demo actions: {demo_actions.shape}")
    print(f"   Current states: {current_states.shape}")
    print(f"   Predicted actions: {predicted_actions.shape}")

    assert predicted_actions.shape == (batch_size, seq_len, action_dim), "Shape mismatch!"
    print("   ✅ ICLTransformerPolicy works!")
    print()

    # Test single-step prediction
    print("2. Testing single-step prediction...")
    single_state = torch.randn(batch_size, state_dim)
    single_action = model.predict_single_step(demo_states, demo_actions, single_state, demo_mask)

    print(f"   Single state: {single_state.shape}")
    print(f"   Predicted action: {single_action.shape}")

    assert single_action.shape == (batch_size, action_dim), "Shape mismatch!"
    print("   ✅ Single-step prediction works!")
    print()

    # Test autoregressive version
    print("3. Testing ICLTransformerPolicyAutoregressive...")
    ar_model = ICLTransformerPolicyAutoregressive(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim
    )

    current_actions = torch.randn(batch_size, seq_len - 1, action_dim)
    ar_predicted_actions = ar_model(
        demo_states, demo_actions, current_states,
        current_actions=current_actions,
        demo_mask=demo_mask
    )

    print(f"   Previous actions: {current_actions.shape}")
    print(f"   Predicted actions: {ar_predicted_actions.shape}")

    assert ar_predicted_actions.shape == (batch_size, seq_len, action_dim), "Shape mismatch!"
    print("   ✅ Autoregressive policy works!")
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 70)
    print(f"✅ All models working!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print("=" * 70)
