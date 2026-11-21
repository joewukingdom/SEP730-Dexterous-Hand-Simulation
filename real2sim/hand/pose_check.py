"""Joint real2sim check: does the same joint angle give the same finger pose in the sim and
on the real hand?

    # 1. write the replay file (a few known poses, each held still)
    python real2sim/hand/pose_check.py make
    # 2. replay it with orca_core (cube removed) while recording both cameras
    #    (record.py posecheck01, then replay_continuous.py ... --replay-file pose_check.yaml)
    # 3. compare: sim hand rendered at exactly those angles over the real frames
    python real2sim/hand/pose_check.py compare posecheck01

Poses (orca_core joint names, degrees; the wrist stays at its neutral value throughout):
  neutral      the config's neutral_position
  zero         every finger joint at 0 (the sim's start pose)
  fingers_45   all MCP and PIP at 45 deg, thumb at 0
  thumb        thumb_cmc 20, thumb_abd 25, thumb_mcp 30, thumb_dip 45, other fingers at 0
  spread       index_abd 20, ring_abd 15, pinky_abd 20, the rest at 0
  neutral      again
Each pose is held HOLD_S seconds, with RAMP_S-second eased moves in between.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ORCA = Path(r"C:\Users\Owner\Desktop\orca")
CONFIG = ORCA / "orca_core" / "models" / "v2" / "orcahand_right" / "config.yaml"
OUT_NAME = "pose_check.yaml"
SCHEDULE = HERE / "pose_check_schedule.json"
HZ, HOLD_S, RAMP_S = 30.0, 4.0, 2.5


def poses(cfg: dict) -> list[tuple[str, dict[str, float]]]:
    ids = cfg["joint_ids"]
    neutral = {j: float(cfg["neutral_position"].get(j, 0.0)) for j in ids}
    wrist = neutral["wrist"]

    def base(**kw):
        p = {j: 0.0 for j in ids}
        p["wrist"] = wrist
        p.update(kw)
        return p

    f45 = {f"{f}_{k}": 45.0 for f in ("index", "middle", "ring", "pinky") for k in ("mcp", "pip")}
    return [
        ("neutral", neutral),
        ("zero", base()),
        ("fingers_45", base(**f45)),
        ("thumb", base(thumb_cmc=20.0, thumb_abd=25.0, thumb_mcp=30.0, thumb_dip=45.0)),
        ("spread", base(index_abd=20.0, ring_abd=15.0, pinky_abd=20.0)),
        ("neutral_end", neutral),
    ]


def make() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ids, roms = cfg["joint_ids"], cfg["joint_roms"]
    seq = poses(cfg)
    vecs = []
    for name, p in seq:
        v = np.array([p[j] for j in ids], float)
        for i, j in enumerate(ids):
            lo, hi = roms[j]
            if not lo <= v[i] <= hi:
                raise SystemExit(f"pose {name}: {j}={v[i]} outside ROM {roms[j]}")
        vecs.append(v)
    n_hold, n_ramp = int(HOLD_S * HZ), int(RAMP_S * HZ)
    s = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n_ramp))[:, None]
    frames, schedule, t = [], [], 0.0
    for k, ((name, _), v) in enumerate(zip(seq, vecs)):
        if k:
            frames.append(vecs[k - 1] + s * (v - vecs[k - 1]))
            t += n_ramp / HZ
        frames.append(np.repeat(v[None], n_hold, axis=0))
        schedule.append({"pose": name, "hold_start_s": round(t, 3), "hold_end_s": round(t + n_hold / HZ, 3),
                         "angles_deg": {j: float(x) for j, x in zip(ids, v)}})
        t += n_hold / HZ
    full = np.vstack(frames)
    peak = (np.abs(np.diff(full, axis=0)) * HZ).max()
    out = ORCA / "replay_sequences" / OUT_NAME
    out.write_text(yaml.safe_dump({
        "metadata": {"type": "continuous", "created_at": time.strftime("%Y%m%d_%H%M%S"),
                     "sampling_frequency_hz": HZ, "joint_ids": ids, "hand_type": cfg["type"],
                     "source": "real2sim pose check (known poses, held)", "record_mode": "commanded"},
        "angles": [[round(float(x), 3) for x in row] for row in full],
    }, sort_keys=False), encoding="utf-8")
    SCHEDULE.write_text(json.dumps({"hz": HZ, "hold_s": HOLD_S, "ramp_s": RAMP_S, "poses": schedule}, indent=2))
    print(f"wrote {out}: {len(full) / HZ:.1f} s, {len(seq)} poses held {HOLD_S:g} s each, "
          f"fastest joint {peak:.0f} deg/s")
    for p in schedule:
        print(f"  {p['hold_start_s']:5.1f}-{p['hold_end_s']:5.1f} s  {p['pose']}")
    print(f"schedule: {SCHEDULE}")


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "make":
        make()
    elif len(sys.argv) >= 3 and sys.argv[1] == "compare":
        raise SystemExit("compare: written once the recording exists")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
