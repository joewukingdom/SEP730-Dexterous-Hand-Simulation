"""Watch a PPO reorientation policy act in the MuJoCo viewer -- also while it is
still training.

    python scripts/watch_policy.py runs/ppo_reorient_random_main
    python scripts/watch_policy.py runs/ppo_reorient_random_main --checkpoint final

Before every episode the newest checkpoint in <run>/checkpoints/ is (re)loaded, so
leaving this window open during training shows the policy improving over time.
Each episode's outcome is printed in the terminal. Close the window to stop.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from orca_sim import OrcaHandRightCubeOrientation  # noqa: E402
from orca_sim.wrappers import RelativeJointTargetAction  # noqa: E402


def newest_checkpoint(run: Path) -> tuple[Path, Path, int] | None:
    best = None
    for zip_path in (run / "checkpoints").glob("ppo_*_steps.zip"):
        m = re.fullmatch(r"ppo_(\d+)_steps\.zip", zip_path.name)
        if not m:
            continue
        steps = int(m.group(1))
        norm = zip_path.with_name(f"ppo_vecnormalize_{steps}_steps.pkl")
        if norm.exists() and (best is None or steps > best[2]):
            best = (zip_path, norm, steps)
    return best


def resolve(run: Path, which: str) -> tuple[Path, Path, str]:
    if which == "final":
        return run / "final_model.zip", run / "vecnormalize.pkl", "final"
    ckpt = newest_checkpoint(run)
    if ckpt is None:
        raise SystemExit(f"No checkpoint with normalisation stats yet in {run / 'checkpoints'} "
                         "(the first one is written after --checkpoint-every steps).")
    return ckpt[0], ckpt[1], f"{ckpt[2]:,} steps"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="training run directory, e.g. runs/ppo_reorient_random_main")
    parser.add_argument("--checkpoint", choices=["latest", "final"], default="latest")
    parser.add_argument("--init", choices=["random", "down"], default=None,
                        help="initial cube orientation (default: taken from the run directory name)")
    parser.add_argument("--episodes", type=int, default=0, help="0 = until the window is closed")
    parser.add_argument("--headless", action="store_true", help="no viewer, just print outcomes (for testing)")
    args = parser.parse_args()

    cfg_path = args.run / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"action": "absolute"}
    init = args.init or cfg.get("init") or ("down" if "_down_" in args.run.name else "random")
    base_env = OrcaHandRightCubeOrientation(initial_red_face=init)
    env = base_env
    if cfg.get("action") == "relative":
        env = RelativeJointTargetAction(base_env, max_delta_frac=cfg.get("max_delta_frac", 0.05))
    vec = DummyVecEnv([lambda: env])
    dt = base_env.model.opt.timestep * base_env.frame_skip

    loaded_from = None
    model = norm = None

    def reload_if_newer() -> None:
        nonlocal loaded_from, model, norm
        zip_path, norm_path, label = resolve(args.run, args.checkpoint)
        if zip_path == loaded_from:
            return
        norm = VecNormalize.load(str(norm_path), vec)
        norm.training = False
        norm.norm_reward = False
        model = PPO.load(str(zip_path), device="cpu")
        loaded_from = zip_path
        print(f"--- loaded policy @ {label}")

    viewer = None if args.headless else mujoco.viewer.launch_passive(base_env.model, base_env.data)
    if viewer is not None:
        mujoco.mjv_defaultFreeCamera(base_env.model, viewer.cam)

    rng = np.random.default_rng()
    ep = 0
    try:
        while (viewer is None or viewer.is_running()) and (args.episodes == 0 or ep < args.episodes):
            ep += 1
            reload_if_newer()
            obs, info = env.reset(seed=int(rng.integers(1 << 31)))
            ret, best_align = 0.0, -1.0
            while True:
                t0 = time.perf_counter()
                action, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
                obs, r, term, trunc, info = env.step(action[0])
                ret += r
                best_align = max(best_align, info["red_face_up_alignment"])
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                if term or trunc:
                    break
            outcome = "SUCCESS" if info["is_success"] else ("dropped" if info["dropped"] else "timeout")
            best_deg = np.degrees(np.arccos(np.clip(best_align, -1, 1)))
            print(f"episode {ep}: {outcome:8s} steps={info['elapsed_steps']:3d} return={ret:7.1f} "
                  f"closest to goal={best_deg:5.1f} deg (need <15 held 25 steps)")
            if viewer is not None:
                time.sleep(0.8)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
