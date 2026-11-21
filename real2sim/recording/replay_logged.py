"""orca_core's replay_continuous.py, plus a time log of every frame sent to the hand.

Run it from the orca folder with its environment, exactly like replay_continuous.py:

    cd C:\\Users\\Owner\\Desktop\\orca
    uv run python "C:\\Users\\Owner\\Desktop\\ORCA project\\SEP730-Dexterous-Hand-Simulation\\real2sim\\recording\\replay_logged.py" ^
        orca_core/models/v2/orcahand_right/config.yaml --replay-file policy_run04.yaml --log run04

The hand is driven the same way (create_hand, connect_hand, init_joints, then
set_joint_positions at the file's rate; Ctrl+C stops). In addition, each frame's
time.perf_counter() is written to real2sim/data/replay_logs/<log>.csv. perf_counter is
the system-wide QueryPerformanceCounter on Windows, the same clock record.py stamps its
samples with, so the replay can be aligned with a recording exactly (twin.py --replay-log).
Standalone on purpose: needs only orca_core (the orca folder's environment), numpy and yaml.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ORCA = Path(r"C:\Users\Owner\Desktop\orca")
LOG_DIR = Path(__file__).resolve().parents[1] / "data" / "replay_logs"
sys.path.insert(0, str(ORCA / "scripts"))
from common import connect_hand, create_hand, resolve_input_path, shutdown_hand  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config_path")
    ap.add_argument("--replay-file", required=True)
    ap.add_argument("--log", required=True, help="name of the time log (real2sim/data/replay_logs/<log>.csv)")
    args = ap.parse_args()

    path = resolve_input_path(args.replay_file)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    meta = data.get("metadata", {})
    if meta.get("type") != "continuous" or "sampling_frequency_hz" not in meta:
        print("Replay file is not a continuous recording with sampling_frequency_hz.")
        return 1
    frames = data.get("angles", [])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{args.log}.csv"
    rows = []
    hand = create_hand(args.config_path, use_mock=False)
    try:
        connect_hand(hand)
        hand.init_joints(force_calibrate=False)
        if meta.get("joint_ids") is not None and meta["joint_ids"] != hand.config.joint_ids:
            raise ValueError("Replay joint order does not match the connected hand configuration.")
        print(f"Replaying {len(frames)} frames from {path}; time log -> {log_path}")
        step = 1.0 / float(meta["sampling_frequency_hz"])
        start = time.time()
        for i, pose in enumerate(frames):
            hand.set_joint_positions(np.asarray(pose, dtype=np.float64))
            rows.append((i, time.perf_counter()))
            remaining = start + i * step - time.time()
            if remaining > 0:
                time.sleep(remaining)
        return 0
    except KeyboardInterrupt:
        print("\nReplay interrupted.")
        return 0
    finally:
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "perf_counter"])
            w.writerows((i, f"{t:.6f}") for i, t in rows)
        print(f"wrote {len(rows)} frame times to {log_path}")
        shutdown_hand(hand)


if __name__ == "__main__":
    raise SystemExit(main())
