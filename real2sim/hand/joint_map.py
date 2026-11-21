"""Joint angles between the sim and the real hand (orca_core), with the corrections measured
on Joe's bench.

orca/sim_to_real.py (JointBridge) already renames thumb_pip <-> thumb_dip and flips the sign of
index_abd. The camera pose checks (docs/real2sim_camera_calibration.md, 8.8) showed that
ring_abd and pinky_abd also run the opposite way, and that the real hand's zero is spread
further out. Here, for those joints:

    sim_deg = SIGN * core_deg + OFFSET_DEG

All other joints use JointBridge unchanged (sim_deg = core_deg, index_abd negated).
Offsets are by-eye estimates (+-5-10 deg) from abdcheck01; abdcheck02 checks them.
"""

from __future__ import annotations

import numpy as np

FIX = {
    "ring_abd": {"sign": -1.0, "offset_deg": -20.0},
    "pinky_abd": {"sign": -1.0, "offset_deg": -27.0},
}


def core_to_sim_deg(joint: str, core_deg: float) -> float:
    if joint in FIX:
        f = FIX[joint]
        return f["sign"] * core_deg + f["offset_deg"]
    return -core_deg if joint == "index_abd" else core_deg


def sim_to_core_deg(joint: str, sim_deg: float) -> float:
    if joint in FIX:
        f = FIX[joint]
        return (sim_deg - f["offset_deg"]) / f["sign"]
    return -sim_deg if joint == "index_abd" else sim_deg


def set_sim_pose(bridge, qpos: np.ndarray, core_pose: dict[str, float]) -> None:
    """Write a real-hand (orca_core) pose into a sim qpos, through the corrected mapping."""
    for j, deg in core_pose.items():
        qpos[bridge.qadr_of_core[j]] = np.radians(core_to_sim_deg(j, deg))


def read_core_degrees(bridge, data) -> list[float]:
    """The sim hand's pose as orca_core angles (core joint order), through the corrected mapping."""
    return [sim_to_core_deg(j, float(np.degrees(data.qpos[bridge.qadr_of_core[j]]))) for j in bridge.core_joint_ids]
