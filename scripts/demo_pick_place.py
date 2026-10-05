"""Watch the scripted pick-and-place policy run live in the MuJoCo viewer.

Opens an interactive window and repeatedly: randomizes the block/bowl
layout, picks up the red block, carries it to the bowl, and releases it.
Prints each episode's outcome to the terminal. Close the viewer window to
stop.

Usage:
    python scripts/demo_pick_place.py
    python scripts/demo_pick_place.py --seed 7 --episodes 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_place_lib import BEST_GRASP, Sim, pick_and_place  # noqa: E402
from validate_pick_place import OBJECT_NAMES, sample_layout  # noqa: E402


class RealtimePacer:
    """Sleeps inside the render callback so playback matches wall-clock
    time instead of running as fast as the physics can step."""

    def __init__(self) -> None:
        self.wall0 = time.perf_counter()
        self.sim0: float | None = None

    def wait(self, sim_time: float) -> None:
        if self.sim0 is None:
            self.sim0 = sim_time
        target = self.wall0 + (sim_time - self.sim0)
        remaining = target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0, help="0 = loop until window is closed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sim = Sim()
    # IMPORTANT: open the viewer on this one Sim and reuse it for every
    # episode (just reset qpos in place below). Re-creating a new Sim()
    # inside the loop would leave the viewer watching a stale, disconnected
    # model that never moves again -- which is exactly what happened before.
    with mujoco.viewer.launch_passive(sim.m, sim.d) as viewer:
        mujoco.mjv_defaultFreeCamera(sim.m, viewer.cam)
        ep = 0
        while viewer.is_running() and (args.episodes == 0 or ep < args.episodes):
            ep += 1
            mujoco.mj_resetDataKeyframe(sim.m, sim.d, 0)
            viewer.sync()
            pts = sample_layout(rng)
            for name, p in zip(OBJECT_NAMES, pts):
                adr = sim.m.jnt_qposadr[sim.m.joint(f"{name}_joint").id]
                sim.d.qpos[adr : adr + 2] = p
            mujoco.mj_forward(sim.m, sim.d)

            pacer = RealtimePacer()

            def render(d: mujoco.MjData) -> None:
                viewer.sync()
                pacer.wait(d.time)

            red_xy = sim.d.body("red_block").xpos[:2].copy()
            bowl_xy = sim.d.body("bowl").xpos[:2].copy()
            outcome = pick_and_place(sim, red_xy, bowl_xy, grasp=BEST_GRASP, render=render)

            red_pos = sim.d.body("red_block").xpos
            bowl_pos = sim.d.body("bowl").xpos
            in_bowl = (
                outcome == "done"
                and np.linalg.norm(red_pos[:2] - bowl_pos[:2]) < 0.055
                and red_pos[2] < bowl_pos[2] + 0.06
            )
            print(f"episode {ep}: {'success' if in_bowl else outcome}")
            time.sleep(0.8)  # pause so you can see the final state before the next episode


if __name__ == "__main__":
    main()
