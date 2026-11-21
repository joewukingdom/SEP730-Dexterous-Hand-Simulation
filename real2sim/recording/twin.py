"""Full digital twin of a policy-replay recording: the sim hand follows the replayed joint
trajectory, the sim cube follows the AprilTag pose, and the sim ultrasonic sees both.

    python real2sim/recording/twin.py run04 policy_run04.yaml run04      # with a replay_logged.py time log (exact)
    python real2sim/recording/twin.py run03 policy_run03.yaml            # without: start guessed, UNRELIABLE

Needs <recording>/dataset.csv (process.py) and the replay file that was played
(orca replay_sequences/<file>). Steps:
  1. replay start in the recording: the offset that best correlates the image motion
     (top camera frame differences) with the commanded joint speed of the replay file;
  2. per dataset row: sim hand at the commanded angles of that moment (through
     real2sim/hand/joint_map.py), sim cube at the camera pose (if seen);
  3. sim ultrasonic (specular model, rig.yaml) with hand AND cube, and which of them the
     echoing ray hit.
Writes <recording>/dataset_twin.csv, docs/figures/real2sim/<name>_twin_timeline.png and
<recording>/twin_overlay.jpg (full sim scene over the real frames at a few moments).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ORCA = Path(r"C:\Users\Owner\Desktop\orca")
sys.path.insert(0, str(HERE.parent / "cameras"))
sys.path.insert(0, str(HERE.parent / "hand"))
sys.path.insert(0, str(ORCA))
import camlib as C  # noqa: E402
from joint_map import set_sim_pose  # noqa: E402
from sim_to_real import JointBridge  # noqa: E402

CONFIG = ORCA / "orca_core" / "models" / "v2" / "orcahand_right" / "config.yaml"
SEES_MM = 150.0
BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(x, np.ones(n) / n, mode="same")


def find_replay_start(rec: Path, angles: np.ndarray, hz: float) -> tuple[float, float]:
    frames = [(float(r["t"]), r["file"]) for r in csv.DictReader(open(rec / "top_frames.csv", newline=""))]
    t = np.array([f[0] for f in frames])
    prev, motion = None, []
    for _, f in frames:
        g = cv2.resize(cv2.imread(str(rec / f), cv2.IMREAD_GRAYSCALE), (240, 135)).astype(np.float32)
        motion.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    motion = smooth(np.array(motion), 5)
    # 1. onset: the hand is still until the replay's first ramp; first sustained motion after a
    #    still first second (whole-recording correlation is fooled by later disturbances, e.g.
    #    a person re-placing the cube)
    rest = motion[t < t[0] + 1.0]
    thr = np.median(rest) + 6 * max(np.median(np.abs(rest - np.median(rest))), 0.05) + 0.5
    above = np.where(motion > thr)[0]
    onset = next((t[i] for i in above if np.all(motion[i:i + 5] > thr)), t[0])
    # 2. refine within [onset - 1, onset + 0.5] by correlating the first 10 s of the replay
    speed = smooth(np.r_[0, np.abs(np.diff(angles, axis=0)).sum(axis=1)], int(0.3 * hz))
    t_rep = np.arange(len(angles)) / hz
    first = t_rep <= 10.0
    best = (-2.0, onset)
    for off in np.arange(onset - 1.0, onset + 0.5, 0.02):
        tq = off + t_rep[first]
        ok = (tq >= t[0]) & (tq <= t[-1])
        if ok.sum() < 20:
            continue
        c = np.corrcoef(np.interp(tq[ok], t, motion), speed[first][ok])[0, 1]
        if c > best[0]:
            best = (c, off)
    print(f"  motion onset at t={onset:.2f} s (threshold {thr:.2f})")
    return best[1], best[0]


def refine_by_pose(rec: Path, angles: np.ndarray, hz: float, ids: list[str], m, d, bridge, guess: float,
                   span: tuple[float, float] = (-6.0, 6.0), step: float = 0.25) -> tuple[float, float]:
    """Replay start by matching the hand itself: for each candidate start, the sim hand at the
    commanded pose is rendered in the top view at ~1 s intervals and its outline compared with
    the real image edges (median distance); the start with the best fit wins."""
    frames = [(float(r["t"]), r["file"]) for r in csv.DictReader(open(rec / "top_frames.csv", newline=""))]
    intr = C.Intrinsics.load(C.intrinsics_path("top"))
    samples = frames[:: max(1, len(frames) // 20)]
    dts = []
    for t, f in samples:
        und = cv2.undistort(cv2.imread(str(rec / f)), intr.K_np, intr.dist_np)
        e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 120)
        dts.append((t, cv2.distanceTransform(255 - e, cv2.DIST_L2, 5)))
    cube = m.body("task_cube").id
    hand = [g for g in range(m.ngeom) if m.geom_bodyid[g] not in (0, cube)]
    cj = m.joint("cube_freejoint").qposadr[0]
    r = mujoco.Renderer(m, 1080, 1920)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    best = (np.inf, guess)
    for off in np.arange(guess + span[0], guess + span[1] + 1e-9, step):
        sc = []
        for t, dist in dts:
            k = int(np.clip(round((t - off) * hz), 0, len(angles) - 1))
            mujoco.mj_resetData(m, d)
            set_sim_pose(bridge, d.qpos, dict(zip(ids, angles[k])))
            d.qpos[cj:cj + 3] = [0.0, 0.0, -1.0]
            mujoco.mj_forward(m, d)
            r.enable_segmentation_rendering()
            r.update_scene(d, camera="top", scene_option=opt)
            seg = r.render()[..., 0]
            r.disable_segmentation_rendering()
            ext, _ = cv2.findContours(np.isin(seg, hand).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pts = np.vstack([c.reshape(-1, 2) for c in ext])
            sc.append(float(np.median(dist[pts[:, 1], pts[:, 0]])))
        s = float(np.mean(sc))
        if s < best[0]:
            best = (s, off)
    return best[1], best[0]


def echo(m, d, cfg: dict, max_inc: float) -> tuple[float | None, str]:
    """Specular sim reading over everything in the scene, and what the echoing ray hit."""
    cube = m.body("task_cube").id
    best, what = None, "none"
    for i in range(m.nsensor):
        if not m.sensor(i).name.startswith("us_"):
            continue
        sid = m.sensor_objid[i]
        o = d.site_xpos[sid].copy()
        R = d.site_xmat[sid].reshape(3, 3)
        v = R[:, 2].copy()
        g = np.array([-1], np.int32)
        t = mujoco.mj_ray(m, d, o, v, None, 1, -1, g)
        if t < 0:
            continue
        p = o + t * v
        pts = []
        for e in (R[:, 0], R[:, 1]):
            g2 = np.array([-1], np.int32)
            t2 = mujoco.mj_ray(m, d, o + 1e-3 * e, v, None, 1, -1, g2)
            if t2 < 0 or g2[0] != g[0]:
                break
            pts.append(o + 1e-3 * e + t2 * v)
        if len(pts) < 2:
            continue
        n = np.cross(pts[0] - p, pts[1] - p)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if np.degrees(np.arccos(min(1.0, abs(float(n @ v))))) > max_inc:
            continue
        if best is None or t < best:
            body = m.geom_bodyid[g[0]]
            best, what = t, ("cube" if body == cube else "bench" if body == 0 else "hand")
    if best is None:
        return None, "none"
    return max(best + cfg.get("range_offset_m", 0.0), cfg.get("min_range_m", 0.0)), what


def main() -> None:
    name, replay_name = sys.argv[1], sys.argv[2]
    rec = C.DATA_DIR / "recordings" / name
    rep = yaml.safe_load((ORCA / "replay_sequences" / replay_name).read_text(encoding="utf-8"))
    angles = np.asarray(rep["angles"], float)
    hz = float(rep["metadata"]["sampling_frequency_hz"])
    ids = rep["metadata"]["joint_ids"]
    log_name = sys.argv[3] if len(sys.argv) >= 4 else None
    exact = None
    if log_name:  # replay_logged.py time log: exact alignment, no guessing
        meta = json.loads((rec / "meta.json").read_text())
        log = list(csv.DictReader(open(C.DATA_DIR / "replay_logs" / f"{log_name}.csv", newline="")))
        exact = np.array([float(r["perf_counter"]) - meta["perf_counter_origin"] for r in log])
        print(f"replay log {log_name}: {len(exact)} frames sent, first at t={exact[0]:.3f} s, "
              f"rate {len(exact) / (exact[-1] - exact[0]):.1f} Hz")
    off0, corr = find_replay_start(rec, angles, hz) if exact is None else (float(exact[0]), float("nan"))
    print(f"replay {replay_name}: {len(angles) / hz:.1f} s; start t={off0:.2f} s "
          f"({'from the replay log' if exact is not None else f'motion-based, corr {corr:.2f}'})")

    rig = C.load_rig(C.HERE / "rig.yaml")
    us_cfg = rig["ultrasonic"]
    max_inc = us_cfg.get("max_incidence_deg", 25.0)
    m = mujoco.MjModel.from_xml_path(str((C.SCENES_DIR / rig["scene"].replace(".xml", "_real2sim.xml")).resolve()))
    d = mujoco.MjData(m)
    bridge = JointBridge(m, yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["joint_ids"])
    cj = m.joint("cube_freejoint").qposadr[0]
    if exact is None:
        # without a log the start is a guess; the pose match below was found NOT to pin it down
        # on run03 (flat score for any start before the recording): treat as unreliable
        off, fit_px = refine_by_pose(rec, angles, hz, ids, m, d, bridge, off0, span=(-8.0, 6.0))
        off, fit_px = refine_by_pose(rec, angles, hz, ids, m, d, bridge, off, span=(-0.5, 0.5), step=0.05)
        print(f"pose-matched start t={off:.2f} s (hand outline fit {fit_px:.1f} px) -- UNRELIABLE without a replay log")
    else:
        off, fit_px = off0, float("nan")

    def frame_at(t: float) -> int:
        if exact is not None:  # the frame actually being shown at time t
            return int(np.clip(np.searchsorted(exact, t, side="right") - 1, 0, len(angles) - 1))
        return int(np.clip(round((t - off) * hz), 0, len(angles) - 1))

    def place(t: float, row: dict) -> int:
        k = frame_at(t)
        mujoco.mj_resetData(m, d)
        set_sim_pose(bridge, d.qpos, dict(zip(ids, angles[k])))
        if row["cube_found"] == "1":
            d.qpos[cj:cj + 3] = [float(row[c]) for c in ("cube_x", "cube_y", "cube_z")]
            d.qpos[cj + 3:cj + 7] = [float(row[c]) for c in ("cube_qw", "cube_qx", "cube_qy", "cube_qz")]
        else:
            d.qpos[cj:cj + 3] = [0.0, 0.0, -1.0]
            d.qpos[cj + 3:cj + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        return k

    rows = list(csv.DictReader(open(rec / "dataset.csv", newline="")))
    out = []
    for r in rows:
        t = float(r["t"])
        k = place(t, r)
        dist, what = echo(m, d, us_cfg, max_inc)
        r2 = dict(r)
        r2["replay_frame"] = k if 0 <= (t - off) * hz <= len(angles) - 1 else ""
        r2["sim_twin_mm"] = "" if dist is None else f"{dist * 1000:.1f}"
        r2["sim_twin_hit"] = what
        r2["sim_twin_sees"] = int(dist is not None and dist * 1000 <= SEES_MM)
        r2["real_sees_any"] = int(bool(r["us_mm"]) and float(r["us_mm"]) <= SEES_MM)
        out.append(r2)
    cols = list(rows[0].keys()) + ["replay_frame", "sim_twin_mm", "sim_twin_hit", "sim_twin_sees", "real_sees_any"]
    with open(rec / "dataset_twin.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)

    during = [r for r in out if r["replay_frame"] != ""]
    agree = sum(r["sim_twin_sees"] == r["real_sees_any"] for r in during)
    both = [(float(r["us_mm"]), float(r["sim_twin_mm"])) for r in during
            if r["real_sees_any"] and r["sim_twin_sees"] and float(r["us_mm"]) >= 25]
    err = np.array([s - u for u, s in both]) if both else np.array([np.nan])
    hits = {h: sum(r["sim_twin_hit"] == h for r in during) for h in ("cube", "hand", "bench", "none")}
    print(f"during the replay ({len(during)} samples): 'something within 150 mm' agreement real vs twin "
          f"{agree}/{len(during)} = {agree / max(len(during), 1):.0%}; twin echo from {hits}; "
          f"both see something (real >= 25 mm): twin - real median {np.nanmedian(err):+.1f} mm, "
          f"MAE {np.nanmean(np.abs(err)):.1f} mm (n={len(both)})")

    # overlay of the full sim scene at a few moments
    intr = {c: C.Intrinsics.load(C.intrinsics_path(c)) for c in ("top", "side")}
    r_ = mujoco.Renderer(m, 1080, 1920)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    pick = [during[int(i)] for i in np.linspace(0, len(during) - 1, 4)] if during else []
    tiles = []
    for r in pick:
        place(float(r["t"]), r)
        for cam in ("top", "side"):
            und = cv2.undistort(cv2.imread(str(rec / r[f"{cam}_file"])), intr[cam].K_np, intr[cam].dist_np)
            r_.enable_segmentation_rendering()
            r_.update_scene(d, camera=cam, scene_option=opt)
            seg = r_.render()[..., 0]
            r_.disable_segmentation_rendering()
            mask = (seg > 0) & ~np.isin(seg, [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0])
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(und, cnts, -1, (255, 0, 255), 3)
            crop = und[100:1000, 600:1500] if cam == "top" else und[250:850, 700:1920]
            crop = cv2.resize(crop, (int(360 * crop.shape[1] / crop.shape[0]), 360))
            cv2.putText(crop, f"t={float(r['t']) - float(rows[0]['t']):.1f}s {cam}", (8, 28), 0, 0.8, (255, 255, 255), 2)
            tiles.append(crop)
    if tiles:
        rows_img = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]
        wmax = max(x.shape[1] for x in rows_img)
        rows_img = [cv2.copyMakeBorder(x, 0, 0, 0, wmax - x.shape[1], cv2.BORDER_CONSTANT) for x in rows_img]
        cv2.imwrite(str(rec / "twin_overlay.jpg"), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 85])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = float(rows[0]["t"])
    tt = np.array([float(r["t"]) - t0 for r in out])

    def series(key, clip=160):
        v = np.array([float(r[key]) if r.get(key) not in ("", None) else np.nan for r in out])
        return np.minimum(v, clip)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), dpi=200, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor(SURF)
    a1.axvspan(off - t0, off - t0 + len(angles) / hz, color="#f0efec", zorder=0)
    a1.plot(tt, series("us_mm"), color=BLUE, lw=2, label="real HC-SR04")
    a1.plot(tt, series("sim_twin_mm"), color=ORANGE, lw=1.4, label="sim twin (hand follows replay + cube)")
    a1.plot(tt, series("sim_specular_mm"), color=AQUA, lw=1.2, ls=(0, (4, 2)), label="sim, cube only")
    a1.set_ylabel("distance (mm)\n(>150: shown at 160)", color=INK, fontsize=9)
    a1.legend(frameon=False, fontsize=8, loc="upper right")
    code = {"none": 0, "bench": 1, "hand": 2, "cube": 3}
    a2.scatter(tt, [code[r["sim_twin_hit"]] for r in out], s=6, color=INK)
    a2.set_yticks([0, 1, 2, 3], ["nothing", "bench", "hand", "cube"])
    a2.set_ylabel("twin echo\nfrom", color=INK, fontsize=9)
    a2.set_xlabel("time (s); shaded: replay running", color=INK, fontsize=9)
    for ax in (a1, a2):
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    a1.set_title(f"{name}: real ultrasonic vs full sim twin", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    figp = C.REPO_ROOT / "docs" / "figures" / "real2sim" / f"{name}_twin_timeline.png"
    fig.savefig(figp, facecolor=SURF)
    print("wrote", rec / "dataset_twin.csv", figp, rec / "twin_overlay.jpg")
    (rec / "twin_meta.json").write_text(json.dumps({
        "replay": replay_name, "replay_start_t": off, "alignment": "replay log" if exact is not None else
        "UNRELIABLE: guessed from image motion / hand pose, no replay log", "replay_log": log_name,
        "motion_start_guess_t": off0, "motion_speed_corr": corr, "pose_fit_px": fit_px}, indent=2))


if __name__ == "__main__":
    main()
