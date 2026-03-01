# Isaac Sim Integration Guide

This guide explains how to use the trained ICL model to control a drone in Isaac Sim.

## Overview

The `scripts/isaac_sim_inference.py` script provides a real-time controller that:
- Loads your trained ICL model
- Maintains demonstration context
- Predicts actions at 50Hz control frequency
- Integrates with Isaac Sim's drone simulation
- **Validates track/demo alignment** to ensure proper performance
- **Provides 3rd person camera view** setup instructions

## Files

- **[scripts/isaac_sim_inference.py](scripts/isaac_sim_inference.py)** - Main inference controller
- **[configs/track_ellipse_example.json](configs/track_ellipse_example.json)** - Example track configuration

## Quick Start

### 1. Prepare Track Configuration

Create a JSON file with your track's gate positions:

```json
{
    "track_name": "ellipse",
    "gates": [
        {"id": 0, "position": [10.0, 0.0, 2.0]},
        {"id": 1, "position": [8.0, 5.0, 2.5]},
        ...
    ],
    "gate_radius": 2.0
}
```

### 2. Test Controller Setup

```bash
python3 scripts/isaac_sim_inference.py \
    --checkpoint checkpoints/ratm_icl_expanded/best_model.pt \
    --data data/ratm_racing_dataset.h5 \
    --track-config configs/track_ellipse_example.json \
    --demo-episodes ep_000001 ep_000002 ep_000003 \
    --device cuda
```

This will:
- Load the model and verify it works
- Load demonstration episodes
- Load track gates
- Print integration instructions

### 3. Isaac Sim Integration Code

In your Isaac Sim Python script:

```python
import numpy as np
from scripts.isaac_sim_inference import (
    IsaacSimICLController,
    load_checkpoint,
    load_demo_episodes,
    load_track_gates
)
from drone_icl.dataset_icl import ICLRacingDataset

# ============================================================
# SETUP (run once at startup)
# ============================================================

# Load model
model, config = load_checkpoint(
    'checkpoints/ratm_icl_expanded/best_model.pt',
    device='cuda'
)

# Load dataset for normalization and demos
dataset = ICLRacingDataset(
    h5_path='data/ratm_racing_dataset.h5',
    num_demos=3,
    max_seq_len=512,
    use_images=False,
    normalize=True,
    subsample_factor=10
)

# Load demo episodes
demo_states, demo_actions, demo_masks = load_demo_episodes(
    ['ep_000001', 'ep_000002', 'ep_000003'],
    dataset
)

# Load track gates
gates, track_name = load_track_gates('configs/track_ellipse_example.json')

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
    device='cuda',
    max_seq_len=512
)

print(f"✅ Controller ready for {track_name} track with {len(gates)} gates")

# ============================================================
# EPISODE LOOP
# ============================================================

def run_episode():
    # Reset controller
    controller.reset()
    current_gate_idx = 0
    gates_passed = 0

    # Isaac Sim setup
    reset_drone_to_start()  # Your Isaac Sim reset function

    # Control loop (runs at 50Hz)
    dt = 1.0 / 50.0  # 50Hz control
    done = False

    while not done:
        # --------------------------------------------------------
        # 1. GET DRONE STATE FROM ISAAC SIM
        # --------------------------------------------------------
        # Get drone pose (position + orientation)
        drone_pose = get_drone_pose()  # (7,) [x, y, z, quat_w, quat_x, quat_y, quat_z]

        # Get drone velocity
        drone_velocity = get_drone_linear_velocity()  # (3,) [vx, vy, vz]

        # Get drone angular velocity
        drone_angular_velocity = get_drone_angular_velocity()  # (3,) [wx, wy, wz]

        # --------------------------------------------------------
        # 2. CONSTRUCT STATE VECTOR
        # --------------------------------------------------------
        state = controller.get_state_from_isaac(
            drone_pose=drone_pose,
            drone_velocity=drone_velocity,
            drone_angular_velocity=drone_angular_velocity,
            gates=gates,
            current_gate_idx=current_gate_idx
        )

        # --------------------------------------------------------
        # 3. PREDICT ACTION
        # --------------------------------------------------------
        action = controller.predict_action(state)
        # action is (4,) [thrust, roll, pitch, yaw]
        # - thrust: [0, 1]
        # - roll, pitch, yaw: [-1, 1]

        # --------------------------------------------------------
        # 4. APPLY ACTION TO DRONE
        # --------------------------------------------------------
        apply_drone_action(action)  # Your Isaac Sim control function

        # --------------------------------------------------------
        # 5. CHECK GATE PASSING
        # --------------------------------------------------------
        if current_gate_idx < len(gates):
            gate_pos = gates[current_gate_idx]
            dist_to_gate = np.linalg.norm(drone_pose[:3] - gate_pos)

            if dist_to_gate < 2.0:  # gate_radius
                current_gate_idx += 1
                gates_passed += 1
                print(f"✅ Gate {gates_passed}/{len(gates)} passed!")

        # --------------------------------------------------------
        # 6. CHECK EPISODE TERMINATION
        # --------------------------------------------------------
        if current_gate_idx >= len(gates):
            print(f"🎉 All gates passed! Episode complete.")
            done = True

        # Check collision or timeout
        if check_collision() or get_episode_time() > 60.0:
            print(f"❌ Episode terminated. Gates passed: {gates_passed}/{len(gates)}")
            done = True

        # Step simulation
        step_simulation(dt)

    return gates_passed


# Run multiple episodes
for episode in range(10):
    print(f"\n{'='*60}")
    print(f"Episode {episode + 1}/10")
    print(f"{'='*60}")
    gates_passed = run_episode()
```

## State Vector Format

The controller expects a 19-dimensional state vector:

| Index | Component | Description |
|-------|-----------|-------------|
| 0-2   | `pos_x, pos_y, pos_z` | Drone position (m) |
| 3-5   | `vel_x, vel_y, vel_z` | Linear velocity (m/s) |
| 6-9   | `quat_w, quat_x, quat_y, quat_z` | Orientation quaternion |
| 10-12 | `ang_vel_x, ang_vel_y, ang_vel_z` | Angular velocity (rad/s) |
| 13-15 | `gate_x, gate_y, gate_z` | Next gate position (m) |
| 16    | `gate_dist` | Distance to next gate (m) |
| 17    | `gate_angle` | Angle to next gate (rad) |
| 18    | `gate_idx` | Current gate index |

## Action Vector Format

The controller outputs a 4-dimensional action vector:

| Index | Component | Range | Description |
|-------|-----------|-------|-------------|
| 0     | `thrust`  | [0, 1] | Upward thrust |
| 1     | `roll`    | [-1, 1] | Roll command |
| 2     | `pitch`   | [-1, 1] | Pitch command |
| 3     | `yaw`     | [-1, 1] | Yaw rate command |

**Note:** You may need to scale/map these actions to your specific drone controller's command format.

## Tips for Isaac Sim Integration

### 1. Frame Rate
- Run the controller at **50Hz** (20ms steps)
- Model was trained on data collected at 50Hz

### 2. Coordinate Systems
- Ensure Isaac Sim uses the same coordinate system as training data
- Typical: X forward, Y left, Z up

### 3. Gate Detection
- Use a radius of **2.0 meters** for gate passing
- Track the current gate index and increment when passed

### 4. Demonstration Episodes
- Use 3 demo episodes from the **same track type** as your test track
- Demo episodes should come from validation set (never seen during training)

### 5. Action Mapping
- The predicted actions are normalized outputs
- You may need to scale them to match your drone's actuator limits:
  ```python
  # Example scaling
  thrust_cmd = action[0] * max_thrust  # e.g., max_thrust = 20.0 N
  roll_rate_cmd = action[1] * max_roll_rate  # e.g., max_roll_rate = 3.0 rad/s
  pitch_rate_cmd = action[2] * max_pitch_rate
  yaw_rate_cmd = action[3] * max_yaw_rate
  ```

### 6. Safety
- Add safety checks for position limits
- Implement emergency stop if drone goes out of bounds
- Monitor velocity limits

### 7. 3rd Person Camera Setup
Enable 3rd person view in Isaac Sim:

```python
# In your Isaac Sim setup
from omni.isaac.core.utils.camera import Camera

# Create 3rd person camera that follows drone
camera = Camera(
    prim_path="/World/Camera",
    position=np.array([0, -5, 2]),  # Behind and above drone
    resolution=(1280, 720)
)

# In control loop, update camera to follow drone
def update_camera(drone_position):
    # Position camera 5m behind and 2m above drone
    camera_offset = np.array([0, -5, 2])  # Adjust as needed
    camera.set_world_pose(
        position=drone_position + camera_offset,
        orientation=np.array([1, 0, 0, 0])  # Look forward
    )

# Call this each frame
update_camera(get_drone_position())
```

**Visualization tips:**
- Display current gate index and gates passed on screen
- Show predicted vs actual trajectory (optional)
- Add velocity/acceleration indicators
- Highlight next gate in different color

## Troubleshooting

### Model predicts erratic actions
- Check state normalization - ensure using same mean/std as training
- Verify gate positions are in correct coordinate frame
- Check that demos are from the same track type

### Drone doesn't move
- Verify action scaling is correct for your drone model
- Check that thrust values are non-zero
- Ensure drone physics are enabled

### Poor performance on new track
- Model only generalizes to track types seen during training (ellipse, lemniscate, trackRATM)
- For truly new track geometries, provide more diverse training data

### Slow inference
- Use GPU (`device='cuda'`)
- Reduce `max_seq_len` if memory is limited
- Batch multiple predictions if running multiple drones

## Performance Expectations

Based on validation results:

| Track Type | Expected Action MSE | Gates Passed (%) |
|------------|-------------------|------------------|
| Ellipse    | 0.07-0.10        | 90-100%          |
| Lemniscate | 0.08-0.12        | 85-95%           |
| TrackRATM  | 0.23-0.35        | 70-85%           |

**Note:** Performance on truly unknown track geometries will be lower. For best results, use tracks similar to training data.

## Contact

For issues with Isaac Sim integration, check:
1. State vector format matches expected (19-dim)
2. Actions are properly scaled for your drone
3. Gate positions are correct in world frame
4. Demo episodes are from validation set

Good luck with your Isaac Sim deployment! 🚁
