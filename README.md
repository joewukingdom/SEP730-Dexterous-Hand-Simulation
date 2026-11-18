<p align="center">
  <img src="https://huggingface.co/datasets/fracapuano/blogs/resolve/main/orca_sim.png" alt="orca_sim header" width="600"/>
</p>


`orca_sim` provides simulation environments for the ORCA hand.
You can start building your ORCA hand today at [orcahand.com](https://www.orcahand.com/).

## Install

We recommend using python 3.11 inside of a virtual environment.
You can create a virtual environment using `uv` ([how to install uv](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
uv venv orca --python 3.11
source orca/bin/activate
uv pip install orca_sim
```

Alternatively, you can use `conda` ([how to install conda](https://www.anaconda.com/docs/getting-started/miniconda/install)):
```bash
conda create -n orca python=3.11 -y
conda activate orca
python -m pip install orca_sim
```
As we are continuously iterating on `orca_sim`, you can fetch the latest `main` building this package from source, so to be in the loop with the latest developments.

```bash
git clone https://github.com/orcahand/orca_sim
cd orca_sim && uv pip install -e .
```

> [!WARNING] 
We are still iterating (a lot!) on this package. If you need stability, consider sticking to the Pypi package (`pip install orca_sim`).

## Demos

Each demo opens a MuJoCo viewer window; close it to stop. Run them from the repository root. The PPO demo also needs `uv pip install stable-baselines3` (plus `matplotlib` for the curves).

| Demo | What you see | Command |
|---|---|---|
| Hands, random actions | Any hand variant moving under random actions | `python random_policy.py --env combined --version v2` |
| Slider control panel | Pick any scene (single hand, two hands, hands on Franka Panda arms, tabletop), drag joint sliders, switch cameras, record and replay poses | `python slider_control.py` |
| Arm pick-and-place | v2 hand on a Franka Panda picks up the red block and puts it in the bowl, on a new random layout every episode (scripted IK, not learned) | `python scripts/demo_pick_place.py` |
| Pick-and-place success rate | Same policy without a viewer; prints the success tally (about 85-90%) | `python scripts/validate_pick_place.py --episodes 20` |
| PPO cube flip | The trained PPO policy turns the cube red face up and holds it (80% from random starts) | `python scripts/watch_policy.py models/ppo_reorient_relative --checkpoint final` |
| PPO training curves | Return, episode length and success rate over the 10M training steps | `python scripts/plot_training.py models/ppo_reorient_relative --once` |

Useful options:

- `random_policy.py --env`: `right`, `left`, `combined`, `right_extended`, `left_extended`, `combined_extended`, `right_cube_orientation`.
- `demo_pick_place.py --seed 7 --episodes 5`: a fixed set of layouts and a set number of episodes.
- `watch_policy.py --init down`: always start with the red face down.
- In `slider_control.py`, click the viewer and press `R` to record a pose, `P` to replay, `S`/`L` to save/load `poses.json`.

The scenes themselves are in [`src/orca_sim/scenes/v2/`](src/orca_sim/scenes/v2/): `scene_right_arm.xml` (one arm), `scene_combined_arm.xml` (two arms) and `scene_right_arm_tabletop.xml` (the pick-and-place table). [`ENVIRONMENTS_MANUAL.md`](ENVIRONMENTS_MANUAL.md) documents every environment.

## Getting started

`orca_sim` follows the [Gymnasium](https://gymnasium.farama.org/) API, and uses [Mujoco](https://mujoco.readthedocs.io/en/stable/overview.html) for physics simulation and rendering.
You can instantiate an environment with one hand (or both) via:

```python
from orca_sim import OrcaHandRight  # or OrcaHandLeft, OrcaHandCombined

env = OrcaHandRight()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```
The 'extended' version of different hands contain additional bodies (incl. inertial properties) such as the camera mount, the U2D2 board and fans.

### Hands versioning

By default, any environment defaults to the latest fully-supported embodiment files.

```python
from orca_sim import OrcaHandRight, OrcaHandRightExtended

env = OrcaHandRight()  # latest version of the standard right hand
extended_env = OrcaHandRightExtended()  # latest version of the extended right hand
```

`orca_sim` stored different hand versions under versioned `scenes/` and `models/` directories. You can still pin an older version explicitly when needed:

```python
from orca_sim import OrcaHandCombinedExtended

env = OrcaHandCombinedExtended(version="v1")  # loads the v1 hand
```

See our [`random_policy.py`](random_policy.py) example to see how to instantiate and interface an ORCA hand.

## Sample task: in-hand cube orientation

`orca_sim` now also ships a task-level example that augments the right hand with a free-floating cube whose one target face is colored red:

```python
from orca_sim import OrcaHandRightCubeOrientation

env = OrcaHandRightCubeOrientation(version="v2", render_mode="human")
obs, info = env.reset(seed=0)
```

By default, the task resets to a palm-up open-hand pose with the cube resting on the palm and the red face pointing downward. You can also randomize the initial cube orientation while keeping it unsolved:

```python
env = OrcaHandRightCubeOrientation(
    version="v2",
    initial_red_face="random",
    cube_pos_xy_jitter=0.01,
)
obs, info = env.reset(seed=0)
```

If you want to keep the environment completely deterministic while you are still building it, you can use the nominal reset directly and add randomization later:

```python
env = OrcaHandRightCubeOrientation(version="v2")

nominal = env.nominal_reset_options()
obs, info = env.reset(options=nominal)

randomized = env.sample_randomized_reset_options(
    seed=0,
    initial_red_face="random",
    cube_pos_xy_jitter=0.01,
)
obs, info = env.reset(options=randomized)
```

The implementation is intentionally split so it doubles as a porting template:

- The task scene lives in [`src/orca_sim/scenes/v2/scene_right_cube_orientation.xml`](src/orca_sim/scenes/v2/scene_right_cube_orientation.xml) and composes the existing hand MJCF with a single task cube.
- The nominal palm-up hand pose and in-palm cube spawn are now authored into the task-specific scene/model files, so opening the XML directly in MuJoCo shows the intended setup.
- The task logic lives in [`src/orca_sim/task_envs.py`](src/orca_sim/task_envs.py), including reset-time cube randomization and optional hand-pose overrides for custom MJCF layouts.

An episode counts as a success only when the red face points up within 15° and the cube is held still (|ω| ≤ 1 rad/s, |v| ≤ 0.1 m/s) for 25 consecutive steps. Dropping the cube ends the episode as a failure. See [`ENVIRONMENTS_MANUAL.md`](ENVIRONMENTS_MANUAL.md) for the full reward and success definition.

## Training a PPO policy on the cube task

The training scripts use [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), which is not a dependency of the package:

```bash
uv pip install stable-baselines3 matplotlib
```

### Watch the pretrained policy

A trained policy is included in [`models/ppo_reorient_relative/`](models/ppo_reorient_relative/). It reaches 80% success from random start orientations (200 episodes), 100% from red-face-down, with no drops. To watch it in the MuJoCo viewer and plot its training curves:

```bash
python scripts/watch_policy.py models/ppo_reorient_relative --checkpoint final
python scripts/plot_training.py models/ppo_reorient_relative --once --save curves.png
```

Each episode's outcome is printed in the terminal. Close the viewer window to stop.

### Train your own

```bash
python scripts/train_ppo_reorient.py
```

The defaults are 10M steps, 16 parallel environments, random start orientations and relative joint control (each action moves the joint targets by at most 5% of their range). The included model took about 1 h 45 min (~1,600 steps/s) on a desktop PC. Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--steps` | `10000000` | total environment steps |
| `--n-envs` | `16` | parallel simulation processes |
| `--init` | `random` | start orientation: `random` or `down` (red face down) |
| `--action` | `relative` | `relative` joint targets, or `absolute` (tends to chatter between joint limits) |
| `--resume` | | continue from a checkpoint `.zip` |
| `--out` | `runs/ppo_reorient_<init>_<timestamp>` | output folder |

The output folder holds `final_model.zip`, `best_model.zip`, `vecnormalize.pkl` (observation normalisation stats, needed to run the policy), `progress.csv` and checkpoints every 500k steps. While training runs, you can follow it from a second terminal:

```bash
python scripts/plot_training.py runs/<run_folder>   # curves, refreshed every 30 s
python scripts/watch_policy.py runs/<run_folder>    # reloads the newest checkpoint each episode
```

