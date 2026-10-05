"""Validate the scripted pick-and-place policy on randomized tabletop layouts.

    python scripts/validate_pick_place.py --episodes 20

Prints a success/fail/ik_fail tally. Failed episodes are expected -- this is
a scripted IK baseline, not a learned policy -- and should simply be
discarded when generating VLA demonstration data.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_place_lib import BEST_GRASP, Sim, pick_and_place  # noqa: E402

OBJECT_NAMES = ["red_block", "bowl", "green_block", "blue_block"]


def sample_layout(rng: np.random.Generator, min_sep: float = 0.12) -> np.ndarray:
    while True:
        pts = np.c_[rng.uniform(0.45, 0.65, 4), rng.uniform(-0.22, 0.22, 4)]
        dists = [np.linalg.norm(pts[i] - pts[j]) for i in range(4) for j in range(i)]
        if min(dists) > min_sep:
            return pts


def run_episode(rng: np.random.Generator) -> str:
    sim = Sim()
    pts = sample_layout(rng)
    for name, p in zip(OBJECT_NAMES, pts):
        adr = sim.m.jnt_qposadr[sim.m.joint(f"{name}_joint").id]
        sim.d.qpos[adr : adr + 2] = p
    mujoco.mj_forward(sim.m, sim.d)

    red_xy = sim.d.body("red_block").xpos[:2].copy()
    bowl_xy = sim.d.body("bowl").xpos[:2].copy()
    outcome = pick_and_place(sim, red_xy, bowl_xy, grasp=BEST_GRASP)
    if outcome == "ik_fail":
        return "ik_fail"

    red_pos = sim.d.body("red_block").xpos
    bowl_pos = sim.d.body("bowl").xpos
    in_bowl = (
        np.linalg.norm(red_pos[:2] - bowl_pos[:2]) < 0.055
        and red_pos[2] < bowl_pos[2] + 0.06
    )
    return "success" if in_bowl else "fail"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results = [run_episode(rng) for _ in range(args.episodes)]
    tally = Counter(results)
    print(f"grasp config: {BEST_GRASP}")
    print(f"{args.episodes} episodes: {dict(tally)}")
    print(f"success rate: {tally['success'] / args.episodes:.0%}")


if __name__ == "__main__":
    main()
