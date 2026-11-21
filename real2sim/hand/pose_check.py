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


def find_offset(t: np.ndarray, motion: np.ndarray, sched: dict) -> tuple[float, float]:
    """Replay start time in the recording: the offset at which the scheduled holds are the
    stillest and the scheduled moves the busiest."""
    still = motion < np.percentile(motion, 40)
    best = (-1e9, 0.0)
    for off in np.arange(t[0] - 5, t[-1] - 30, 0.05):
        sc, n = 0.0, 0
        for p in sched["poses"][1:]:  # the first hold blends with the rest before the replay
            a, b = off + p["hold_start_s"] + 0.5, off + p["hold_end_s"] - 0.5
            m = (t >= a) & (t <= b)
            r = (t >= off + p["hold_start_s"] - sched["ramp_s"] + 0.3) & (t <= off + p["hold_start_s"] - 0.3)
            if m.any() and r.any():
                sc += still[m].mean() - still[r].mean()
                n += 1
        if n and sc / n > best[0]:
            best = (sc / n, off)
    return best[1], best[0]


def compare(name: str) -> None:
    import csv

    import cv2
    import mujoco

    sys.path.insert(0, str(HERE.parent / "cameras"))
    sys.path.insert(0, str(ORCA))
    import camlib as C
    from sim_to_real import JointBridge

    rec = C.DATA_DIR / "recordings" / name
    sched = json.loads(SCHEDULE.read_text())
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    frames = {c: [(float(r["t"]), r["file"]) for r in csv.DictReader(open(rec / f"{c}_frames.csv", newline=""))]
              for c in ("top", "side")}
    # motion energy of the top view (downscaled frame differences)
    t_top = np.array([t for t, _ in frames["top"]])
    prev, motion = None, []
    for _, f in frames["top"]:
        g = cv2.resize(cv2.imread(str(rec / f), cv2.IMREAD_GRAYSCALE), (240, 135)).astype(np.float32)
        motion.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    motion = np.array(motion)
    off, score = find_offset(t_top, motion, sched)
    print(f"replay starts at t={off:.2f} s in the recording (stillness contrast {score:.2f})")

    rig = C.load_rig(C.HERE / "rig.yaml")
    m = mujoco.MjModel.from_xml_path(str((C.SCENES_DIR / rig["scene"].replace(".xml", "_real2sim.xml")).resolve()))
    d = mujoco.MjData(m)
    bridge = JointBridge(m, cfg["joint_ids"])
    cube = m.body("task_cube").id
    cj = m.joint("cube_freejoint").qposadr[0]
    hand_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] not in (0, cube)}
    r = mujoco.Renderer(m, 1080, 1920)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    cal = json.loads((C.CALIB_DIR / "cameras.json").read_text())
    rows, report = [], []
    for p in sched["poses"]:
        t_mid = off + (p["hold_start_s"] + p["hold_end_s"]) / 2
        mujoco.mj_resetData(m, d)
        for j, deg in p["angles_deg"].items():
            d.qpos[bridge.qadr_of_core[j]] = np.radians(-deg if j == "index_abd" else deg)
        d.qpos[cj:cj + 3] = [0.0, 0.0, -1.0]  # cube out of the way
        mujoco.mj_forward(m, d)
        tiles = []
        for cam in ("top", "side"):
            t_f, f = min(frames[cam], key=lambda x: abs(x[0] - t_mid))
            intr = C.Intrinsics.load(C.intrinsics_path(cam))
            und = cv2.undistort(cv2.imread(str(rec / f)), intr.K_np, intr.dist_np)
            r.enable_segmentation_rendering()
            r.update_scene(d, camera=cam, scene_option=opt)
            seg = r.render()[..., 0]
            r.disable_segmentation_rendering()
            mask = np.isin(seg, list(hand_geoms)).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            vis = und.copy()
            cv2.drawContours(vis, cnts, -1, (255, 0, 255), 3)
            ys, xs = np.nonzero(mask)
            x0, x1 = max(0, xs.min() - 120), min(1920, xs.max() + 120)
            y0, y1 = max(0, ys.min() - 120), min(1080, ys.max() + 120)
            crop = vis[y0:y1, x0:x1]
            crop = cv2.resize(crop, (int(540 * crop.shape[1] / crop.shape[0]), 540))
            cv2.putText(crop, f"{p['pose']} - {cam}", (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 7)
            cv2.putText(crop, f"{p['pose']} - {cam}", (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
            tiles.append(crop)
        w = sum(t.shape[1] for t in tiles)
        rows.append(cv2.resize(np.hstack(tiles), (1800, int(540 * 1800 / w))))
    sheet = np.vstack(rows)
    out = rec / "pose_check.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    fig = C.REPO_ROOT / "docs" / "figures" / "real2sim" / f"{name}_pose_check.jpg"
    cv2.imwrite(str(fig), cv2.resize(sheet, (1500, int(sheet.shape[0] * 1500 / sheet.shape[1]))),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {out} and {fig}")


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "make":
        make()
    elif len(sys.argv) >= 3 and sys.argv[1] == "compare":
        compare(sys.argv[2])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
