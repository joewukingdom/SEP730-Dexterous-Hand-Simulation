# ORCA Sim — Environments Manual

## Overview

Every environment follows the **Gymnasium** interface (`reset` / `step` / `close`).  
All environments come in two hand versions: **v1** and **v2** (default is v2).  
Actions and observations are NumPy arrays.

---

## Versions

| Version | Default? | Notes |
|---------|----------|-------|
| `v1` | No | 17 actuators per hand; simpler joint set |
| `v2` | **Yes** | 17 actuators per hand; finer control ranges, extra thumb CMC joint |

Pass `version="v1"` or `version="v2"` to any constructor to select the version.

---

## Environment List

### 1. `OrcaHandRight` / `OrcaHandRight-v1` / `OrcaHandRight-v2`

Single **right** hand, no object.

```python
from orca_sim import OrcaHandRight

env = OrcaHandRight(render_mode="human", version="v1")
obs, info = env.reset()
```

- **Action space:** `Box(17,)` — one value per actuator (radians, position control)
- **Observation:** `qpos` + `qvel` concatenated
- **Use for:** general right-hand motion, teleoperation, policy training

---

### 2. `OrcaHandLeft` / `OrcaHandLeft-v1` / `OrcaHandLeft-v2`

Single **left** hand, no object.

```python
from orca_sim import OrcaHandLeft

env = OrcaHandLeft(render_mode="human", version="v2")
obs, info = env.reset()
```

- **Action space:** `Box(17,)` — same structure as right hand
- **Observation:** `qpos` + `qvel` concatenated
- **Use for:** general left-hand motion

---

### 3. `OrcaHandCombined` / `OrcaHandCombined-v1` / `OrcaHandCombined-v2`

Both hands together in one scene.

```python
from orca_sim import OrcaHandCombined

env = OrcaHandCombined(render_mode="human", version="v2")
obs, info = env.reset()
```

- **Action space:** `Box(34,)` — left 17 + right 17
- **Observation:** `qpos` + `qvel` concatenated (both hands)
- **Use for:** bimanual tasks, dual-hand coordination

---

### 4. `OrcaHandRightExtended` / `OrcaHandRightExtended-v1` / `OrcaHandRightExtended-v2`

Right hand with an **extended arm/wrist workspace** — larger scene, more range of motion.

```python
from orca_sim import OrcaHandRightExtended

env = OrcaHandRightExtended(render_mode="human", version="v2")
obs, info = env.reset()
```

- **Action space:** `Box(17,)` — same as `OrcaHandRight`
- **Use for:** tasks needing a larger arm range or a different mount position

---

### 5. `OrcaHandLeftExtended` / `OrcaHandLeftExtended-v1` / `OrcaHandLeftExtended-v2`

Left hand with extended workspace.

```python
from orca_sim import OrcaHandLeftExtended

env = OrcaHandLeftExtended(render_mode="human", version="v2")
obs, info = env.reset()
```

- **Action space:** `Box(17,)` — same as `OrcaHandLeft`

---

### 6. `OrcaHandCombinedExtended` / `OrcaHandCombinedExtended-v1` / `OrcaHandCombinedExtended-v2`

Both hands, extended workspace.

```python
from orca_sim import OrcaHandCombinedExtended

env = OrcaHandCombinedExtended(render_mode="human", version="v2")
obs, info = env.reset()
```

- **Action space:** `Box(34,)` — same as `OrcaHandCombined`

---

### 7. `OrcaHandRightCubeOrientation` / `OrcaHandRightCubeOrientation-v1` / `OrcaHandRightCubeOrientation-v2`

Right hand holding a **cube** with a red face. Goal: rotate the cube so the red face points **up**.

```python
from orca_sim import OrcaHandRightCubeOrientation

env = OrcaHandRightCubeOrientation(render_mode="human", version="v2")
obs, info = env.reset()
```

- **Action space:** `Box(17,)` — right hand only
- **Observation:** `qpos` + `qvel` + `red_face_world_normal (3,)` + `red_face_up_alignment (1,)` 
- **Reward:** alignment reward (0–1) + small lift bonus − drop penalty
- **Terminated when:** red face is ≤15° from pointing up, OR cube is dropped below 5 cm
- **Truncated when:** episode exceeds `max_episode_steps` (default 200)

#### Extra constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_red_face` | `"down"` | Starting orientation: `"down"` or `"random"` |
| `cube_pos_xy_jitter` | `0.0` | Random XY offset for cube spawn (meters) |
| `max_episode_steps` | `200` | Steps before truncation |
| `success_tolerance_rad` | `0.2618` (15°) | How close to "up" counts as success |
| `drop_height` | `0.05` | Z threshold below which cube is considered dropped |
| `hand_pose_by_joint` | `None` | Dict of `{joint_name: radians}` for custom initial hand pose |

#### `info` dict keys returned by `step()`

| Key | Type | Description |
|-----|------|-------------|
| `cube_pos` | `(3,)` | Current cube XYZ position |
| `cube_quat` | `(4,)` | Current cube quaternion (w, x, y, z) |
| `cube_qvel` | `(6,)` | Cube linear + angular velocity |
| `red_face_world_normal` | `(3,)` | World-frame direction the red face points |
| `red_face_up_alignment` | `float` | Dot product with world-up (1.0 = solved) |
| `red_face_up_angle_rad` | `float` | Angle from solved (0.0 = solved) |
| `is_success` | `bool` | True when goal is reached |
| `dropped` | `bool` | True when cube fell below drop height |
| `elapsed_steps` | `int` | Steps taken in current episode |

#### Custom reset options

```python
# Nominal reset (deterministic, no jitter)
opts = env.nominal_reset_options()
obs, info = env.reset(options=opts)

# Randomized reset (random cube orientation + optional XY jitter)
opts = env.sample_randomized_reset_options(
    seed=42,
    initial_red_face="random",
    cube_pos_xy_jitter=0.01,
)
obs, info = env.reset(options=opts)

# Full manual override
obs, info = env.reset(options={
    "hand_pose_by_joint": {"right_thumb_mcp": 0.5, "right_index_mcp": 0.8},
    "cube_pos": [0.0, 0.0, 0.15],
    "cube_quat": [1.0, 0.0, 0.0, 0.0],
    "settle_steps": 20,
})
```

---

## Using Gymnasium `gym.make`

All environments are also registered in the Gymnasium registry:

```python
import gymnasium as gym
import orca_sim  # triggers register_envs()

env = gym.make("OrcaHandRight-v1", render_mode="human")
env = gym.make("OrcaHandCombined-v2", render_mode="human")
env = gym.make("OrcaHandRightCubeOrientation-v2", render_mode="human")
```

List all registered IDs:

```python
from orca_sim import list_versions
print(list_versions())  # ('v1', 'v2')
```

---

## Render Modes

| Mode | Behaviour |
|------|-----------|
| `"human"` | Opens the MuJoCo interactive viewer. Use the **Ctrl** tab in the right panel for built-in actuator sliders. |
| `"rgb_array"` | Returns a `(H, W, 3)` uint8 NumPy array each step. No window. |
| `None` | No rendering at all. Fastest. |

---

## Common Usage Pattern

```python
import time
import numpy as np
from orca_sim import OrcaHandRight

env = OrcaHandRight(render_mode="human", version="v1")
obs, info = env.reset(seed=0)

for _ in range(500):
    action = env.action_space.sample()          # replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1 / 30)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## v1 Right Hand Actuators Reference

| Index | Name | Range (rad) |
|-------|------|-------------|
| 0 | right_wrist | −0.6736 → 0.8972 |
| 1 | right_thumb_mcp | −0.8727 → 0.8727 |
| 2 | right_thumb_abd | −1.0821 → 0.0 |
| 3 | right_thumb_pip | −0.7944 → 1.23 |
| 4 | right_thumb_dip | −0.8538 → 1.45 |
| 5 | right_index_abd | −1.0458 → 0.2458 |
| 6 | right_index_mcp | −0.3491 → 1.6581 |
| 7 | right_index_pip | −0.3491 → 1.885 |
| 8 | right_middle_abd | −0.6458 → 0.6458 |
| 9 | right_middle_mcp | −0.3491 → 1.5883 |
| 10 | right_middle_pip | −0.3491 → 1.8675 |
| 11 | right_ring_abd | −0.4758 → 0.8058 |
| 12 | right_ring_mcp | −0.3491 → 1.5883 |
| 13 | right_ring_pip | −0.3491 → 1.8675 |
| 14 | right_pinky_abd | −0.1224 → 1.1691 |
| 15 | right_pinky_mcp | −0.3491 → 1.7104 |
| 16 | right_pinky_pip | −0.3491 → 1.885 |
