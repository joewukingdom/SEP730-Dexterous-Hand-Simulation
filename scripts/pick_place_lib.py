"""Scripted pick-and-place helper for the ORCA-on-Panda tabletop scene.

This is the reference policy used to validate ``scene_right_arm_tabletop.xml``
and to generate demonstration trajectories for VLA fine-tuning (OpenVLA,
pi0, ...). It is a plain IK + fixed-grasp-pose controller, not a learned
policy: given a target object, it computes a top-down approach, closes the
hand to a pre-tuned grasp pose, lifts, carries, and releases.

The grasp pose below (BEST_GRASP) was found with a grid search over wrist
pitch, approach depth, thumb pose, and finger curl, evaluated by attempting a
full pick-and-place across 5 randomized block/bowl layouts per candidate.
See scripts/grasp_search.py. It reaches ~85-90% success (failures are stuck
layouts where the bowl lands at an awkward reach, or a knock during carry);
demo generation should simply discard failed episodes.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

import sys

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orca_sim.kinematics import SiteIK  # noqa: E402
from orca_sim.versions import resolve_scene_path  # noqa: E402

SCENE_PATH = resolve_scene_path("scene_right_arm_tabletop.xml", version="v2")

# (wrist_pitch_deg, approach_depth_m, thumb_cmc, thumb_abd, finger_curl)
BEST_GRASP = dict(pitch_deg=55.0, dz=0.045, t_cmc=0.0, t_abd=0.0, curl=0.65)

ARM_HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# In the right_tower (hand mount) frame, the fingers extend along +y and the
# palm faces -x; this is the orientation (expressed as a world-frame rotation
# matrix for the grasp site) for a palm-down approach with fingers along
# world +x. `hand_R(pitch)` tilts that approach down by `pitch`.
R_PALM_DOWN = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)


def hand_R(pitch: float) -> np.ndarray:
    """Approach-site rotation matrix for a given downward wrist pitch (rad)."""
    return rot_y(pitch) @ R_PALM_DOWN


class Sim:
    """Thin wrapper around an MjModel/MjData pair for the tabletop scene,
    with IK and a parametric hand pose for the scripted grasp."""

    def __init__(self) -> None:
        self.m = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.hand_act = np.arange(7, 24)
        self.hand_names = [m.joint(m.actuator_trnid[i, 0]).name for i in self.hand_act]
        self.ik_solver = SiteIK(
            m,
            self.d,
            "right_grasp_site",
            dof_ids=np.arange(7),
            joint_range=m.jnt_range[:7],
            home_qpos=ARM_HOME,
        )
        mujoco.mj_resetDataKeyframe(m, self.d, 0)
        mujoco.mj_forward(m, self.d)

    def ik(self, pos, R, q0=None, iters=300):
        q, pos_err, rot_err = self.ik_solver.solve(pos, R, q0=q0, iters=iters)
        return q, pos_err, rot_err

    def hand_pose(self, curl=0.0, t_cmc=0.0, t_abd=0.0, t_curl=None):
        """17-dim hand ctrl: all finger MCP/PIP joints driven to `curl` of
        their range, thumb CMC/abduction held at fixed values."""
        q = np.zeros(17)
        for i, n in enumerate(self.hand_names):
            lo, hi = self.m.actuator_ctrlrange[self.hand_act[i]]
            if n == "right_t-cmc":
                q[i] = t_cmc
            elif n == "right_t-abd":
                q[i] = t_abd
            elif n.startswith("right_t-"):
                q[i] = hi * (curl if t_curl is None else t_curl)
            elif "mcp" in n or "pip" in n:
                q[i] = hi * curl
            q[i] = np.clip(q[i], lo, hi)
        return q

    def run(self, arm_q=None, hand_q=None, duration_s=1.0, render=None, chunks=20):
        """Linearly ramp ctrl from its current value to the given target(s)
        over `duration_s` seconds of sim time, then hold.

        Without a render callback, physics is advanced in large C-side
        batches (``mujoco.mj_step(..., nstep=...)``) rather than one Python
        call per timestep -- this matters a lot for batch demo generation
        and grid search, where thousands of short rollouts are run.
        """
        d, m = self.d, self.m
        c0 = d.ctrl.copy()
        c1 = c0.copy()
        if arm_q is not None:
            c1[:7] = arm_q
        if hand_q is not None:
            c1[7:] = hand_q
        n = max(1, int(duration_s / m.opt.timestep))
        if render is None:
            ramp_n = max(1, int(0.8 * n))
            seg = max(1, ramp_n // chunks)
            done = 0
            for i in range(chunks):
                steps = seg if i < chunks - 1 else max(0, ramp_n - seg * (chunks - 1))
                if steps <= 0:
                    continue
                alpha = min(1.0, (done + steps) / ramp_n)
                d.ctrl[:] = (1 - alpha) * c0 + alpha * c1
                mujoco.mj_step(m, d, nstep=steps)
                done += steps
            d.ctrl[:] = c1
            remaining = n - done
            if remaining > 0:
                mujoco.mj_step(m, d, nstep=remaining)
        else:
            for k in range(n):
                alpha = min(1.0, (k + 1) / (0.8 * n))
                d.ctrl[:] = (1 - alpha) * c0 + alpha * c1
                mujoco.mj_step(m, d)
                if k % int(0.05 / m.opt.timestep) == 0:
                    render(d)


def pick_and_place(
    sim: Sim,
    pick_xy: np.ndarray,
    place_xy: np.ndarray,
    *,
    grasp: dict = BEST_GRASP,
    render=None,
) -> str:
    """Run the full scripted pick-and-place from `pick_xy` to `place_xy`
    (table-plane coordinates; z is resolved from the current object pose).
    Returns "success", "fail", or "ik_fail"."""
    d, m = sim.d, sim.m
    pitch = np.deg2rad(grasp["pitch_deg"])
    dz, t_cmc, t_abd, curl = grasp["dz"], grasp["t_cmc"], grasp["t_abd"], grasp["curl"]
    R = hand_R(pitch)

    mujoco.mj_forward(m, d)

    open_q = sim.hand_pose(0.0, t_cmc, t_abd)
    close_q = sim.hand_pose(curl, t_cmc, t_abd)

    pick_pos = np.array([pick_xy[0], pick_xy[1], pick_xy[2] if len(pick_xy) > 2 else 0.421])
    place_pos = np.array([place_xy[0], place_xy[1], place_xy[2] if len(place_xy) > 2 else 0.421])
    pick_tgt = pick_pos + [0, 0, dz]
    place_tgt = place_pos + [0, 0, dz]

    q_pre, e, _ = sim.ik(pick_tgt + [0, 0, 0.10], R, iters=400)
    if e > 5e-3:
        return "ik_fail"
    d.qpos[:7] = q_pre
    d.ctrl[:7] = q_pre
    d.qpos[7:24] = open_q
    d.ctrl[7:] = open_q
    mujoco.mj_forward(m, d)

    waypoints = [
        (None, None, 0.3),
        (pick_tgt, None, 1.0),
        (None, close_q, 1.0),
        (pick_tgt + [0, 0, 0.15], None, 1.0),
        (place_tgt + [0, 0, 0.15], None, 1.5),
        (place_tgt + [0, 0, 0.07], None, 0.8),
        (None, open_q, 0.8),
        (place_tgt + [0, 0, 0.18], None, 1.0),
        (None, None, 0.5),
    ]
    q = q_pre
    for pos, hand_q, duration in waypoints:
        arm_q = None
        if pos is not None:
            q, e, _ = sim.ik(pos, R, q0=q, iters=400)
            if e > 5e-3:
                return "ik_fail"
            arm_q = q
        sim.run(arm_q, hand_q, duration, render)
    return "done"
