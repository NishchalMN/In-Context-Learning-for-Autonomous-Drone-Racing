"""
Simulate actual drone flight using PyBullet physics with trained ICL model.

Generates video of:
- 3rd person camera view showing drone flying through gates
- Optional FPV (first-person view) from drone camera
- Real physics simulation using PyBullet
"""

import numpy as np
import torch
from pathlib import Path
import argparse
import h5py

# PyBullet simulation
import pybullet as p
import pybullet_data
from PIL import Image
import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_icl.models import ICLTransformerPolicy


class DroneSimulator:
    """Simple quadrotor simulator using PyBullet."""

    def __init__(self, gui=True, dt=0.02):
        """
        Args:
            gui: Show GUI window
            dt: Timestep (50Hz = 0.02s)
        """
        self.dt = dt

        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Create simple drone (box with 4 propellers)
        self.drone_id = self._create_drone()

        self.gates = []

    def _create_drone(self):
        """Create a simple drone using primitives."""
        # Main body (box)
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05],
                                          rgbaColor=[0.2, 0.2, 0.8, 1.0])

        drone_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 1]
        )

        return drone_id

    def add_gate(self, position, size=1.0):
        """
        Add a racing gate at specified position.

        Args:
            position: [x, y, z] position
            size: Gate size (radius)
        """
        # Create gate as 4 cylinders forming a square
        gate_parts = []

        # Gate posts (vertical cylinders)
        post_positions = [
            [position[0] - size/2, position[1], position[2]],
            [position[0] + size/2, position[1], position[2]],
        ]

        for pos in post_positions:
            collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=size*1.5)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=size*1.5,
                                        rgbaColor=[1, 0, 0, 1])

            gate_part = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=pos
            )
            gate_parts.append(gate_part)

        # Top bar (horizontal cylinder)
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=size)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=size,
                                    rgbaColor=[1, 0, 0, 1])

        top_bar = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[position[0], position[1], position[2] + size*0.75],
            baseOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0])
        )
        gate_parts.append(top_bar)

        self.gates.append(gate_parts)

        return gate_parts

    def get_state(self):
        """
        Get current drone state.

        Returns:
            state: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13D
        """
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)

        state = np.concatenate([
            np.array(pos),
            np.array(quat),
            np.array(lin_vel),
            np.array(ang_vel)
        ])

        return state

    def set_state(self, position, orientation=None, lin_vel=None, ang_vel=None):
        """Set drone state."""
        if orientation is None:
            orientation = [0, 0, 0, 1]  # Identity quaternion

        p.resetBasePositionAndOrientation(self.drone_id, position, orientation)

        if lin_vel is not None and ang_vel is not None:
            p.resetBaseVelocity(self.drone_id, lin_vel, ang_vel)

    def apply_action(self, action):
        """
        Apply control action.

        Args:
            action: [thrust, roll_rate, pitch_rate, yaw_rate]
        """
        thrust = action[0]
        roll_rate = action[1]
        pitch_rate = action[2]
        yaw_rate = action[3]

        # Simple control: apply force and torque
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)

        # Thrust (upward force in body frame)
        force = [0, 0, thrust * 10.0]  # Scale thrust
        p.applyExternalForce(self.drone_id, -1, force, pos, p.WORLD_FRAME)

        # Angular velocities (torques)
        torque = [roll_rate * 0.5, pitch_rate * 0.5, yaw_rate * 0.5]
        p.applyExternalTorque(self.drone_id, -1, torque, p.WORLD_FRAME)

    def step(self):
        """Step simulation forward."""
        p.stepSimulation()

    def render_camera(self, camera_position, target_position, width=640, height=480):
        """
        Render camera view.

        Args:
            camera_position: [x, y, z] camera position
            target_position: [x, y, z] point camera looks at
            width, height: Image resolution

        Returns:
            rgb_array: (H, W, 3) numpy array
        """
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=[0, 0, 1]
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0
        )

        # Get camera image
        img = p.getCameraImage(
            width, height,
            view_matrix,
            projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Extract RGB
        rgb = np.array(img[2], dtype=np.uint8)
        rgb = rgb.reshape((height, width, 4))[:, :, :3]

        return rgb

    def get_fpv(self, width=640, height=480):
        """Get first-person view from drone."""
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)

        # Camera offset from drone center (forward and up)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        camera_offset = rot_matrix @ np.array([0.15, 0, 0.05])
        camera_pos = np.array(pos) + camera_offset

        # Look forward
        forward_offset = rot_matrix @ np.array([1.0, 0, 0])
        target_pos = camera_pos + forward_offset

        return self.render_camera(camera_pos, target_pos, width, height)

    def get_third_person_view(self, width=640, height=480, distance=3.0, angle=30.0):
        """Get third-person view following drone."""
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)

        # Camera behind and above drone
        angle_rad = np.deg2rad(angle)
        camera_pos = [
            pos[0] - distance * np.cos(angle_rad),
            pos[1],
            pos[2] + distance * np.sin(angle_rad)
        ]

        return self.render_camera(camera_pos, pos, width, height)

    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect(self.client)


def load_demonstration_from_dataset(h5_path, episode_idx=0):
    """
    Load demonstration episode for in-context learning.

    Returns:
        demo_states, demo_actions, gates, initial_state
    """
    with h5py.File(h5_path, 'r') as f:
        episodes_group = f['episodes']
        episode_names = sorted([k for k in episodes_group.keys() if k.startswith('ep_')])

        ep_name = episode_names[episode_idx]
        ep = episodes_group[ep_name]

        # Detect format: original has 'actions' dataset, augmented has 'action' subgroup
        is_original_format = 'actions' in ep

        if is_original_format:
            # Original format with 'actions' dataset
            pos = ep['state/pos'][:]
            quat = ep['state/quat'][:]
            lin_vel = ep['state/lin_vel'][:]
            ang_vel = ep['state/ang_vel'][:]
            lin_acc = ep['imu/lin_acc'][:]
            imu_ang_vel = ep['imu/ang_vel'][:]
            actions = ep['actions'][:]
            gates = ep['track_gates'][:]
        else:
            # Augmented format with 'action' subgroup
            pos = ep['state']['pos'][:]
            quat = ep['state']['quat'][:]
            lin_vel = ep['state']['lin_vel'][:]
            ang_vel = ep['state']['ang_vel'][:]
            lin_acc = ep['imu']['lin_acc'][:]
            imu_ang_vel = ep['imu']['ang_vel'][:]

            # Reconstruct actions from thrust + omega
            thrust = ep['action']['thrust'][:]
            omega = ep['action']['omega'][:]
            if thrust.ndim == 1:
                thrust = thrust[:, None]
            actions = np.concatenate([thrust, omega], axis=-1)

            gates = ep['track_gates'][:]

        # Initial state (first timestep)
        initial_state = {
            'position': pos[0],
            'orientation': quat[0],
            'lin_vel': lin_vel[0],
            'ang_vel': ang_vel[0]
        }

        return pos, quat, lin_vel, ang_vel, lin_acc, imu_ang_vel, actions, gates, initial_state


def simulate_with_model(model, demo_states, demo_actions, gates, initial_state,
                        sim_steps=500, device='cuda', use_gate_info=True,
                        state_mean=None, state_std=None, action_mean=None, action_std=None):
    """
    Run simulation using trained model.

    Args:
        model: Trained ICL model
        demo_states: Demo trajectory states for ICL (already normalized)
        demo_actions: Demo trajectory actions (already normalized)
        gates: Gate positions
        initial_state: Initial drone state
        sim_steps: Number of simulation steps
        state_mean, state_std: Normalization stats for states
        action_mean, action_std: Normalization stats for actions

    Returns:
        trajectory: Dict with positions, actions, frames
    """
    # Create simulator
    sim = DroneSimulator(gui=False, dt=0.02)

    # Add gates
    for gate_pos in gates:
        sim.add_gate(gate_pos, size=1.0)

    # Set initial state
    sim.set_state(
        initial_state['position'],
        initial_state['orientation'],
        initial_state['lin_vel'],
        initial_state['ang_vel']
    )

    # Prepare demo for model (convert to tensor)
    # For ICL, we need demo in format: (1, num_demos, T, state_dim)
    # Use full demo as context
    demo_states_tensor = torch.FloatTensor(demo_states).unsqueeze(0).unsqueeze(0).to(device)
    demo_actions_tensor = torch.FloatTensor(demo_actions).unsqueeze(0).unsqueeze(0).to(device)

    # Create demo mask (all ones for valid timesteps)
    demo_mask = torch.ones(1, 1, demo_states.shape[0], dtype=torch.bool, device=device)

    # Storage
    positions = []
    actions_taken = []
    third_person_frames = []
    fpv_frames = []

    # Track previous velocities for IMU computation
    prev_lin_vel = np.array(initial_state['lin_vel'])
    prev_ang_vel = np.array(initial_state['ang_vel'])

    model.eval()

    for step in range(sim_steps):
        # Get current state
        state_13d = sim.get_state()

        # Extract current velocities
        current_lin_vel = state_13d[7:10]
        current_ang_vel = state_13d[10:13]

        # Compute IMU data (accelerations) from velocity changes
        dt = 0.02  # 50Hz
        lin_acc = (current_lin_vel - prev_lin_vel) / dt
        imu_ang_vel = current_ang_vel  # IMU angular velocity is same as state angular velocity

        # Update previous velocities
        prev_lin_vel = current_lin_vel
        prev_ang_vel = current_ang_vel

        # Build 19D state
        state_19d = np.concatenate([state_13d, lin_acc, imu_ang_vel])

        # Add gate features if enabled
        if use_gate_info:
            pos = state_13d[:3]
            # Find nearest gate
            distances = np.linalg.norm(gates - pos, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_gate = gates[nearest_idx]

            # Gate features
            relative_pos = nearest_gate - pos
            distance = np.linalg.norm(relative_pos)
            direction = relative_pos / (distance + 1e-6)

            gate_features = np.concatenate([direction, [distance], relative_pos])
            state = np.concatenate([state_19d, gate_features])
        else:
            state = state_19d

        # Normalize state if normalization stats provided
        if state_mean is not None and state_std is not None:
            state = (state - state_mean.cpu().numpy()) / state_std.cpu().numpy()

        # Predict action using model
        with torch.no_grad():
            current_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, state_dim)

            # Predict (with demo mask)
            action_tensor = model.predict_single_step(
                demo_states_tensor,
                demo_actions_tensor,
                current_state_tensor,
                demo_mask=demo_mask
            )

            action = action_tensor.cpu().numpy()[0]

        # Denormalize action if normalization stats provided
        if action_mean is not None and action_std is not None:
            action = action * action_std.cpu().numpy() + action_mean.cpu().numpy()

        # Apply action
        sim.apply_action(action)
        sim.step()

        # Record
        pos, _ = p.getBasePositionAndOrientation(sim.drone_id)
        positions.append(pos)
        actions_taken.append(action)

        # Render frames every N steps
        if step % 2 == 0:
            third_person_frames.append(sim.get_third_person_view(width=640, height=480))
            fpv_frames.append(sim.get_fpv(width=640, height=480))

    sim.close()

    return {
        'positions': np.array(positions),
        'actions': np.array(actions_taken),
        'third_person_frames': third_person_frames,
        'fpv_frames': fpv_frames,
        'gates': gates
    }


def create_video(frames, output_path, fps=25):
    """Create video from frames using OpenCV."""
    if len(frames) == 0:
        print("No frames to save!")
        return

    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Saved video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Simulate drone flight with PyBullet')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset for demo trajectories')
    parser.add_argument('--episode', type=int, default=2,
                       help='Episode index to use for demo')
    parser.add_argument('--output-dir', type=str, default='simulation_videos',
                       help='Output directory')
    parser.add_argument('--steps', type=int, default=None,
                       help='Number of simulation steps (default: use full demo length)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--use-gate-info', action='store_true', default=False,
                       help='Use gate information (26D state). Set this if model was trained with --use-gate-info')
    parser.add_argument('--fps', type=int, default=25,
                       help='Video FPS')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print("  PyBullet Drone Flight Simulation")
    print("="*70)
    print()

    # Load model
    state_dim = 26 if args.use_gate_info else 19
    print(f"Loading model (state_dim={state_dim})...")

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    embed_dim = checkpoint.get('embed_dim', 256)

    model = ICLTransformerPolicy(
        state_dim=state_dim,
        action_dim=4,
        embed_dim=embed_dim
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded from epoch {checkpoint['epoch']}")
    print()

    # Load demonstration
    print(f"Loading demonstration from episode {args.episode}...")
    pos, quat, lin_vel, ang_vel, lin_acc, imu_ang_vel, actions, gates, initial_state = \
        load_demonstration_from_dataset(args.data, args.episode)

    # Prepare demo states (19D: pos, quat, lin_vel, ang_vel, lin_acc, imu_ang_vel)
    demo_states = np.concatenate([pos, quat, lin_vel, ang_vel, lin_acc, imu_ang_vel], axis=-1)

    print(f"  Demo length: {len(demo_states)} steps")
    print(f"  Gates: {len(gates)}")
    print()

    # Load normalization statistics from dataset
    print("Computing normalization statistics from dataset...")
    from drone_icl.dataset_icl import ICLRacingDataset
    temp_dataset = ICLRacingDataset(
        h5_path=args.data,
        num_demos=1,
        use_gate_info=args.use_gate_info,
        normalize=True
    )
    state_mean = temp_dataset.state_mean.to(args.device)
    state_std = temp_dataset.state_std.to(args.device)
    action_mean = temp_dataset.action_mean.to(args.device)
    action_std = temp_dataset.action_std.to(args.device)
    print(f"  State dim: {len(state_mean)}")
    print(f"  Action dim: {len(action_mean)}")
    print()

    # Normalize demo states and actions
    demo_states_normalized = (demo_states - state_mean.cpu().numpy()) / state_std.cpu().numpy()
    actions_normalized = (actions - action_mean.cpu().numpy()) / action_std.cpu().numpy()

    # Use demo length if steps not specified, otherwise use args.steps
    sim_steps = len(demo_states) if args.steps is None else args.steps
    print(f"Running simulation for {sim_steps} steps (demo length: {len(demo_states)})...")

    trajectory = simulate_with_model(
        model, demo_states_normalized, actions_normalized, gates, initial_state,
        sim_steps=sim_steps,
        device=args.device,
        use_gate_info=args.use_gate_info,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std
    )
    print()

    # Save videos
    print("Creating videos...")
    create_video(
        trajectory['third_person_frames'],
        output_dir / 'third_person.mp4',
        fps=args.fps
    )

    create_video(
        trajectory['fpv_frames'],
        output_dir / 'fpv.mp4',
        fps=args.fps
    )

    print()
    print("="*70)
    print("  Simulation Complete!")
    print("="*70)
    print(f"Videos saved to: {output_dir}")
    print(f"  - third_person.mp4: Third-person chase camera")
    print(f"  - fpv.mp4: First-person view from drone")
    print()


if __name__ == '__main__':
    main()
