"""With the wrist frozen, which starting cube orientation makes the policy's fingers act on the
cube the most? Runs the PPO in the sim for each green-face (sim red face) direction and
reports how far the cube moves and turns, and whether it is dropped.

    python real2sim/policy/orientation_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from orca_sim import OrcaHandRightCubeOrientation  # noqa: E402
from orca_sim.wrappers import RelativeJointTargetAction  # noqa: E402

MODEL = REPO / "models" / "ppo_reorient_relative"
# world direction of the green face -> rotation taking the cube's +z to it
FACES = {"up (+z)": (0, 0, 1), "down (-z)": (0, 0, -1), "fingertips (+x)": (1, 0, 0), "wrist (-x)": (-1, 0, 0),
         "pinky (+y)": (0, 1, 0), "thumb (-y)": (0, -1, 0)}


def quat_z_to(v) -> np.ndarray:
    v = np.asarray(v, float)
    z = np.array([0, 0, 1.0])
    axis = np.cross(z, v)
    s, c = np.linalg.norm(axis), float(z @ v)
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1, 0, 0])
    ang = np.arctan2(s, c)
    axis /= s
    return np.r_[np.cos(ang / 2), np.sin(ang / 2) * axis]


def main() -> None:
    run_cfg = json.loads((MODEL / "run_config.json").read_text())
    base = OrcaHandRightCubeOrientation(initial_red_face="random")
    env = RelativeJointTargetAction(base, max_delta_frac=run_cfg.get("max_delta_frac", 0.05))
    norm = VecNormalize.load(str(MODEL / "vecnormalize.pkl"), DummyVecEnv([lambda: env]))
    norm.training, norm.norm_reward = False, False
    model = PPO.load(str(MODEL / "final_model.zip"), device="cpu")
    wrist = int(mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wrist_actuator")) \
        if mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wrist_actuator") >= 0 else None
    if wrist is None:  # find the actuator driving the wrist joint
        jw = base.model.joint("right_wrist").id
        wrist = int(np.where(base.model.actuator_trnid[:, 0] == jw)[0][0])
    cube = base.model.body("task_cube").id
    dt = base.model.opt.timestep * base.frame_skip
    print(f"{'green face':18s} {'moved (mm)':>10s} {'turned (deg)':>12s} {'goal':>6s} {'dropped':>8s}")
    for label, v in FACES.items():
        obs, _ = env.reset(seed=0, options={"cube_quat": quat_z_to(v)})
        p0 = base.data.xpos[cube].copy()
        R0 = base.data.xmat[cube].reshape(3, 3).copy()
        goal, dropped = None, False
        for k in range(int(2.0 / dt)):
            a, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
            a = np.asarray(a).reshape(-1).copy()
            a[wrist] = 0.0
            obs, _, term, trunc, info = env.step(a)
            if goal is None and info.get("goal_reached"):
                goal = (k + 1) * dt
            if info.get("dropped"):
                dropped = True
                break
        p1 = base.data.xpos[cube]
        R1 = base.data.xmat[cube].reshape(3, 3)
        turned = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
        print(f"{label:18s} {np.linalg.norm(p1 - p0) * 1000:10.1f} {turned:12.1f} "
              f"{'%.2fs' % goal if goal else '-':>6s} {str(dropped):>8s}")


if __name__ == "__main__":
    main()
