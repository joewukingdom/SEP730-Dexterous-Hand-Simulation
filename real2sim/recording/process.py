"""Turn a synchronized recording into the dataset table: per ultrasonic sample the matched
camera frames, the cube pose from the AprilTags (both cameras fused), and what the sim
ultrasonic sensor reads for that cube pose (two echo models).

    python real2sim/recording/process.py run01

Writes real2sim/data/recordings/<name>/dataset.csv and docs/figures/real2sim/<name>_timeline.png.

Columns of dataset.csv:
  t, us_mm, top_file, top_dt_ms, side_file, side_dt_ms, frames_ok   (from align.py)
  cube_found, tags_top, tags_side                                   tags used per camera
  cube_x, cube_y, cube_z (m), cube_qw, cube_qx, cube_qy, cube_qz      world (sim) frame; cube +z = green face
  green_up_deg        angle between the green face (sim red face) normal and world up
  face_incidence_deg  angle between the ultrasonic beam and the cube face that faces it
  sim_nearest_mm      sim: nearest hit in the 15 deg cone (calibrated offset / dead zone)
  sim_specular_mm     sim: same, but a ray only echoes if it meets the face within
                      rig.yaml ultrasonic.max_incidence_deg (25) of its normal (empty = no echo)
  real_sees_cube, sim_nearest_sees_cube, sim_specular_sees_cube     reading <= 150 mm
The sim rays only see the cube (the sim hand is in its default pose, not the real one).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cameras"))
sys.path.insert(0, str(HERE.parent / "cube"))
import camlib as C  # noqa: E402
import cube_tags as CT  # noqa: E402

SEES_MM = 150.0
BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main() -> None:
    name = sys.argv[1]
    rec = C.DATA_DIR / "recordings" / name
    if not (rec / "aligned.csv").exists():
        subprocess.run([sys.executable, str(HERE / "align.py"), name], check=True)
    rows = list(csv.DictReader(open(rec / "aligned.csv", newline="")))
    layout = json.loads(CT.LAYOUT.read_text())
    cal = json.loads((C.CALIB_DIR / "cameras.json").read_text())
    rig = C.load_rig(C.HERE / "rig.yaml")
    us_cfg = rig["ultrasonic"]
    beam = np.asarray(us_cfg["dir"], float)
    beam /= np.linalg.norm(beam)
    intr = {c: C.Intrinsics.load(C.intrinsics_path(c)) for c in ("top", "side")}
    T_wc = {c: np.asarray(cal["cameras"][c]["T_world_cam_opencv"]) for c in ("top", "side")}
    det = CT.detector()

    scene = (C.SCENES_DIR / rig["scene"].replace(".xml", "_real2sim.xml")).resolve()
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    j = m.joint("cube_freejoint").qposadr[0]
    cube_body = m.body("task_cube").id
    hand_bodies = {b for b in range(1, m.nbody) if b != cube_body}

    out, prev_R, prev_t = [], None, -1e9
    cache: dict[str, np.ndarray] = {}

    def img(f):
        if f not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[f] = cv2.imread(str(rec / f))
        return cache[f]

    for k, r in enumerate(rows):
        t = float(r["t"])
        views = [(img(r["top_file"]), intr["top"], T_wc["top"]), (img(r["side_file"]), intr["side"], T_wc["side"])]
        prior = prev_R if t - prev_t < 0.5 else None
        T, info = CT.cube_pose_fused(views, layout, det, prior_R_world_cube=prior)
        row = {**r, "frames_ok": r["ok"]}
        row.pop("ok")
        row["cube_found"] = int(T is not None)
        row["tags_top"] = " ".join(map(str, info.get(0, ([],))[0])) if T is not None else ""
        row["tags_side"] = " ".join(map(str, info.get(1, ([],))[0])) if T is not None else ""
        if T is not None:
            prev_R, prev_t = T[:3, :3], t
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, T[:3, :3].flatten())
            normals = np.hstack([T[:3, :3], -T[:3, :3]])  # the six outward face normals
            facing = normals[:, int(np.argmin(normals.T @ beam))]
            d.qpos[j:j + 3] = T[:3, 3]
            d.qpos[j + 3:j + 7] = q
            mujoco.mj_forward(m, d)
            near = C.ultrasonic_reading_specular(m, d, us_cfg, 90.0, hand_bodies)
            spec = C.ultrasonic_reading_specular(m, d, us_cfg, us_cfg.get("max_incidence_deg", 25.0), hand_bodies)
            row.update({
                "cube_x": f"{T[0, 3]:.4f}", "cube_y": f"{T[1, 3]:.4f}", "cube_z": f"{T[2, 3]:.4f}",
                "cube_qw": f"{q[0]:.4f}", "cube_qx": f"{q[1]:.4f}", "cube_qy": f"{q[2]:.4f}", "cube_qz": f"{q[3]:.4f}",
                "green_up_deg": f"{np.degrees(np.arccos(np.clip(T[2, 2], -1, 1))):.1f}",
                "face_incidence_deg": f"{np.degrees(np.arccos(np.clip(-facing @ beam, -1, 1))):.1f}",
                "sim_nearest_mm": "" if near is None else f"{near * 1000:.1f}",
                "sim_specular_mm": "" if spec is None else f"{spec * 1000:.1f}",
            })
        else:
            row.update({c: "" for c in ("cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz",
                                        "green_up_deg", "face_incidence_deg", "sim_nearest_mm", "sim_specular_mm")})

        def sees(v):
            return "" if v == "" else int(float(v) <= SEES_MM)
        row["real_sees_cube"] = sees(r["us_mm"]) if r["us_mm"] else 0
        row["sim_nearest_sees_cube"] = sees(row["sim_nearest_mm"]) if row["cube_found"] else ""
        row["sim_specular_sees_cube"] = (sees(row["sim_specular_mm"]) if row["sim_specular_mm"] else 0) \
            if row["cube_found"] else ""
        out.append(row)
        if k % 50 == 0:
            print(f"  {k}/{len(rows)}", flush=True)

    cols = ["t", "us_mm", "top_file", "top_dt_ms", "side_file", "side_dt_ms", "frames_ok", "cube_found",
            "tags_top", "tags_side", "cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz",
            "green_up_deg", "face_incidence_deg", "sim_nearest_mm", "sim_specular_mm", "real_sees_cube",
            "sim_nearest_sees_cube", "sim_specular_sees_cube"]
    with open(rec / "dataset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in out:
            w.writerow({c: row.get(c, "") for c in cols})

    found = [r for r in out if r["cube_found"]]
    print(f"\n{rec / 'dataset.csv'}: {len(out)} rows, cube found in {len(found)}")
    for model in ("nearest", "specular"):
        key = f"sim_{model}_sees_cube"
        agree = sum(int(r["real_sees_cube"] == r[key]) for r in found)
        both = [(float(r["us_mm"]), float(r[f"sim_{model}_mm"])) for r in found
                if r["real_sees_cube"] == 1 and r[key] == 1]
        err = np.array([s - u for u, s in both]) if both else np.array([np.nan])
        tab = {(a, b): sum(1 for r in found if r["real_sees_cube"] == a and r[key] == b) for a in (0, 1) for b in (0, 1)}
        print(f"  sim {model:8s}: sees-cube agreement {agree}/{len(found)} "
              f"(real sees & sim sees {tab[(1, 1)]}, real sees & sim not {tab[(1, 0)]}, "
              f"real not & sim sees {tab[(0, 1)]}, neither {tab[(0, 0)]}); when both see it: "
              f"sim - real median {np.nanmedian(err):+.1f} mm, MAE {np.nanmean(np.abs(err)):.1f} mm (n={len(both)})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = float(out[0]["t"])
    tt = np.array([float(r["t"]) - t0 for r in out])

    def series(key, clip=None):
        v = np.array([float(r[key]) if r.get(key) not in ("", None) else np.nan for r in out])
        return np.minimum(v, clip) if clip else v
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), dpi=200, sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(SURF)
    a1.plot(tt, series("us_mm", 160), color=BLUE, lw=2, label="real HC-SR04")
    a1.plot(tt, series("sim_nearest_mm", 160), color=ORANGE, lw=1.4, label="sim, nearest in cone")
    a1.plot(tt, series("sim_specular_mm", 160), color=AQUA, lw=1.4, ls=(0, (4, 2)), label=f"sim, specular (<= {us_cfg.get('max_incidence_deg', 25):g} deg)")
    a1.axhline(SEES_MM, color=INK2, lw=0.8, ls=":")
    a1.text(tt[-1], SEES_MM + 2, "readings above 150 mm: not the cube (shown at 160)", ha="right", va="bottom",
            fontsize=8, color=INK2)
    a1.set_ylabel("distance (mm)", color=INK, fontsize=9)
    a1.legend(frameon=False, fontsize=8, loc="center left")
    a2.plot(tt, series("face_incidence_deg"), color=INK, lw=1.4)
    th = us_cfg.get("max_incidence_deg", 25)
    a2.axhline(th, color=INK2, lw=0.8, ls=":")
    a2.text(tt[0], th + 1, f"{th:g} deg", fontsize=8, color=INK2)
    a2.set_ylabel("beam vs facing\ncube face (deg)", color=INK, fontsize=9)
    a2.set_xlabel("time (s)", color=INK, fontsize=9)
    for ax in (a1, a2):
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    a1.set_title(f"{name}: ultrasonic, real vs sim, from the cube pose seen by the cameras", loc="left",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig_path = C.REPO_ROOT / "docs" / "figures" / "real2sim" / f"{name}_timeline.png"
    fig.savefig(fig_path, facecolor=SURF)
    print("wrote", fig_path)


if __name__ == "__main__":
    main()
