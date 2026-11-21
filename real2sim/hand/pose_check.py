"""Joint real2sim check: does the same joint angle give the same finger pose in the sim and
on the real hand?

    # 1. write the replay file (a few known poses, each held still)
    python real2sim/hand/pose_check.py make
    # 2. replay it with orca_core (cube removed) while recording both cameras
    #    (record.py posecheck01, then replay_continuous.py ... --replay-file pose_check.yaml)
    # 3. compare: sim hand rendered at exactly those angles over the real frames
    python real2sim/hand/pose_check.py compare posecheck01

    # abduction check: index / ring / pinky abd at -20, 0, +20 deg one at a time, then a fit of
    # the per-finger correction that best matches the top view
    python real2sim/hand/pose_check.py make abd          # -> pose_check_abd.yaml
    python real2sim/hand/pose_check.py compare abdcheck01 abd

    # check the corrected mapping (joint_map.py): targets in sim degrees, commanded through it
    python real2sim/hand/pose_check.py make abdfix
    python real2sim/hand/pose_check.py compare abdcheck02 abdfix

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
HZ, HOLD_S, RAMP_S = 30.0, 4.0, 2.5
# pose sets: "basic" (above) and "abd": index / ring / pinky abduction at -20, 0, +20 deg one
# finger at a time, everything else at 0 (index is the reference: it looked aligned)
ABD_JOINTS, ABD_VALUES = ("index_abd", "ring_abd", "pinky_abd"), (-20.0, 20.0)


def out_name(pose_set: str) -> str:
    return "pose_check.yaml" if pose_set == "basic" else f"pose_check_{pose_set}.yaml"


def schedule_path(pose_set: str) -> Path:
    return HERE / ("pose_check_schedule.json" if pose_set == "basic" else f"pose_check_{pose_set}_schedule.json")


def poses(cfg: dict, pose_set: str = "basic") -> list[tuple[str, dict[str, float]]]:
    ids = cfg["joint_ids"]
    neutral = {j: float(cfg["neutral_position"].get(j, 0.0)) for j in ids}
    wrist = neutral["wrist"]

    def base(**kw):
        p = {j: 0.0 for j in ids}
        p["wrist"] = wrist
        p.update(kw)
        return p

    if pose_set == "abdfix":
        # targets given in SIM degrees, commanded through the corrected mapping (joint_map.py):
        # if the correction is right, the sim outline lands on the real finger in every pose
        from joint_map import sim_to_core_deg
        seq = [("neutral", neutral), ("zero", base())]
        for j, targets in (("ring_abd", (-25.0, -10.0, 5.0)), ("pinky_abd", (-28.0, -15.0, 0.0))):
            for s in targets:
                seq.append((f"{j}_sim{s:+.0f}", base(**{j: round(sim_to_core_deg(j, s), 3)})))
            seq.append((f"zero_after_{j}", base()))
        return seq + [("neutral_end", neutral)]
    if pose_set == "abd":
        seq = [("neutral", neutral), ("zero", base())]
        for j in ABD_JOINTS:
            for v in ABD_VALUES:
                seq.append((f"{j}_{v:+.0f}", base(**{j: v})))
            seq.append((f"zero_after_{j}", base()))
        return seq + [("neutral_end", neutral)]
    f45 = {f"{f}_{k}": 45.0 for f in ("index", "middle", "ring", "pinky") for k in ("mcp", "pip")}
    return [
        ("neutral", neutral),
        ("zero", base()),
        ("fingers_45", base(**f45)),
        ("thumb", base(thumb_cmc=20.0, thumb_abd=25.0, thumb_mcp=30.0, thumb_dip=45.0)),
        ("spread", base(index_abd=20.0, ring_abd=15.0, pinky_abd=20.0)),
        ("neutral_end", neutral),
    ]


def make(pose_set: str = "basic") -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ids, roms = cfg["joint_ids"], cfg["joint_roms"]
    seq = poses(cfg, pose_set)
    OUT_NAME, SCHEDULE = out_name(pose_set), schedule_path(pose_set)
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


def compare(name: str, pose_set: str = "basic") -> None:
    import csv

    import cv2
    import mujoco

    sys.path.insert(0, str(HERE.parent / "cameras"))
    sys.path.insert(0, str(ORCA))
    import camlib as C
    from joint_map import set_sim_pose
    from sim_to_real import JointBridge

    rec = C.DATA_DIR / "recordings" / name
    sched = json.loads(schedule_path(pose_set).read_text())
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
        if pose_set == "abdfix":  # render through the corrected mapping being tested
            set_sim_pose(bridge, d.qpos, p["angles_deg"])
        else:
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
            # agreement score: median distance (px) from the sim outline to the nearest real image
            # edge, on the outer outline of the fingers only (below the wrist the hand is fixed)
            edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 120)
            dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
            ext, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pts = np.vstack([c.reshape(-1, 2) for c in ext])
            score = float(np.median(dist[pts[:, 1], pts[:, 0]]))
            report.append((p["pose"], cam, score))
            vis = und.copy()
            cv2.drawContours(vis, cnts, -1, (255, 0, 255), 3)
            ys, xs = np.nonzero(mask)
            x0, x1 = max(0, xs.min() - 120), min(1920, xs.max() + 120)
            y0, y1 = max(0, ys.min() - 120), min(1080, ys.max() + 120)
            crop = vis[y0:y1, x0:x1]
            crop = cv2.resize(crop, (int(540 * crop.shape[1] / crop.shape[0]), 540))
            label = f"{p['pose']} - {cam}  {score:.1f}px"
            cv2.putText(crop, label, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 7)
            cv2.putText(crop, label, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
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
    print("median distance sim outline -> real edges (px, lower = closer):")
    for pose_name, cam, sc in report:
        print(f"  {pose_name:12s} {cam:5s} {sc:5.1f}")
    (rec / "pose_check_scores.json").write_text(json.dumps(
        {"replay_offset_s": off, "scores_px": [{"pose": a, "cam": b, "px": c} for a, b, c in report]}, indent=2))
    if pose_set != "abd":
        return

    # --- abduction offsets: for each finger, in its own +-20 poses and in the zero poses, find
    # the correction (deg, added to the commanded angle) that best fits that finger's outline in
    # the top view. Same correction at -20 / 0 / +20 -> zero offset; correction ~ -2x command ->
    # the joint's direction is reversed between sim and real.
    def finger_geoms(joint: str) -> set[int]:
        qadr = bridge.qadr_of_core[joint]
        jid = int(np.where(m.jnt_qposadr == qadr)[0][0])
        root = int(m.jnt_bodyid[jid])
        bodies = set()
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p == root:
                    bodies.add(b)
                    break
                p = int(m.body_parentid[p])
        return {g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies}

    intr = C.Intrinsics.load(C.intrinsics_path("top"))
    deltas = np.arange(-30, 31, 2.0)
    fit = {}
    print("\nabduction fit (top view, that finger's outline only): best correction to the commanded angle")
    for joint in ABD_JOINTS:
        geoms = finger_geoms(joint)
        for p in sched["poses"]:
            cmd = p["angles_deg"][joint]
            others_zero = all(abs(v) < 1e-6 for k, v in p["angles_deg"].items() if k not in ("wrist", joint))
            if not others_zero:
                continue
            t_mid = off + (p["hold_start_s"] + p["hold_end_s"]) / 2
            _, f = min(frames["top"], key=lambda x: abs(x[0] - t_mid))
            und = cv2.undistort(cv2.imread(str(rec / f)), intr.K_np, intr.dist_np)
            edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 120)
            dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
            scores = []
            for dl in deltas:
                mujoco.mj_resetData(m, d)
                for j, deg in p["angles_deg"].items():
                    v = deg + (dl if j == joint else 0.0)
                    d.qpos[bridge.qadr_of_core[j]] = np.radians(-v if j == "index_abd" else v)
                d.qpos[cj:cj + 3] = [0.0, 0.0, -1.0]
                mujoco.mj_forward(m, d)
                r.enable_segmentation_rendering()
                r.update_scene(d, camera="top", scene_option=opt)
                seg = r.render()[..., 0]
                r.disable_segmentation_rendering()
                mask = np.isin(seg, list(geoms)).astype(np.uint8)
                ext, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not ext:
                    scores.append(np.inf)
                    continue
                pts = np.vstack([c.reshape(-1, 2) for c in ext])
                scores.append(float(np.median(dist[pts[:, 1], pts[:, 0]])))
            k = int(np.argmin(scores))
            fit.setdefault(joint, []).append({"pose": p["pose"], "commanded": cmd, "best_correction": float(deltas[k]),
                                              "score_px": scores[k], "score_at_0_px": scores[len(deltas) // 2]})
            print(f"  {joint:10s} {p['pose']:22s} commanded {cmd:+5.0f}: best correction {deltas[k]:+5.0f} deg "
                  f"({scores[len(deltas) // 2]:.1f} -> {scores[k]:.1f} px)")
    (rec / "abd_fit.json").write_text(json.dumps(fit, indent=2))
    print(f"wrote {rec / 'abd_fit.json'}")


def main() -> None:
    pose_set = sys.argv[3] if len(sys.argv) >= 4 else "basic"
    if len(sys.argv) >= 2 and sys.argv[1] == "make":
        make(sys.argv[2] if len(sys.argv) >= 3 else "basic")
    elif len(sys.argv) >= 3 and sys.argv[1] == "compare":
        compare(sys.argv[2], pose_set)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
