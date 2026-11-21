"""Does the ultrasonic sensor move with the wrist? Track it in both cameras and compare its
image motion with the commanded wrist angle of a logged replay.

    python real2sim/recording/sensor_track.py run06 policy_run06.yaml run06

The sensor's image position comes from rig.yaml (ultrasonic.pos_m) projected into each
camera; a patch around it in the first frame is template-matched in every later frame.
Prints the patch displacement (px) against the wrist angle and their correlation, and
writes docs/figures/real2sim/<name>_sensor_track.png.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ORCA = Path(r"C:\Users\Owner\Desktop\orca")
sys.path.insert(0, str(HERE.parent / "cameras"))
import camlib as C  # noqa: E402

BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main() -> None:
    name, replay_name, log_name = sys.argv[1], sys.argv[2], sys.argv[3]
    rec = C.DATA_DIR / "recordings" / name
    rig = C.load_rig(C.HERE / "rig.yaml")
    cal = json.loads((C.CALIB_DIR / "cameras.json").read_text())
    meta = json.loads((rec / "meta.json").read_text())
    rep = yaml.safe_load((ORCA / "replay_sequences" / replay_name).read_text(encoding="utf-8"))
    w_idx = rep["metadata"]["joint_ids"].index("wrist")
    wrist = np.asarray(rep["angles"], float)[:, w_idx]
    log = np.array([float(r["perf_counter"]) - meta["perf_counter_origin"]
                    for r in csv.DictReader(open(C.DATA_DIR / "replay_logs" / f"{log_name}.csv", newline=""))])
    pos = np.asarray(rig["ultrasonic"]["pos_m"], float)
    out = {}
    for cam in ("top", "side"):
        T = np.asarray(cal["cameras"][cam]["T_world_cam_opencv"])
        intr = C.Intrinsics.load(C.intrinsics_path(cam))
        Tcw = C.inv_T(T)
        uv, _ = cv2.projectPoints(pos[None], cv2.Rodrigues(Tcw[:3, :3])[0], Tcw[:3, 3], intr.K_np, intr.dist_np)
        u, v = uv.ravel().astype(int)
        frames = [(float(r["t"]), r["file"]) for r in csv.DictReader(open(rec / f"{cam}_frames.csv", newline=""))]
        h = 45
        g0 = cv2.imread(str(rec / frames[0][1]), cv2.IMREAD_GRAYSCALE)
        tpl = g0[v - h:v + h, u - h:u + h]
        ts, dx, dy, score = [], [], [], []
        for t, f in frames[::2]:
            g = cv2.imread(str(rec / f), cv2.IMREAD_GRAYSCALE)
            win = g[v - 3 * h:v + 3 * h, u - 3 * h:u + 3 * h]
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            ts.append(t)
            dx.append(loc[0] - 2 * h)
            dy.append(loc[1] - 2 * h)
            score.append(mx)
        ts, d = np.array(ts), np.hypot(dx, dy)
        k = np.clip(np.searchsorted(log, ts, side="right") - 1, 0, len(wrist) - 1)
        wr = np.where(ts < log[0], wrist[0], wrist[k])
        c = np.corrcoef(d, wr)[0, 1] if np.std(d) > 0 and np.std(wr) > 0 else float("nan")
        out[cam] = (ts, d, wr)
        print(f"{cam}: sensor patch displacement median {np.median(d):.1f} px, max {d.max():.1f} px "
              f"(match score median {np.median(score):.2f}); wrist {wr.min():.1f}..{wr.max():.1f} deg; "
              f"corr(displacement, wrist) {c:+.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), dpi=200, sharex=True)
    fig.patch.set_facecolor(SURF)
    t0 = out["top"][0][0]
    axes[0].plot(out["top"][0] - t0, out["top"][2], color=INK, lw=1.6)
    axes[0].set_ylabel("commanded\nwrist (deg)", fontsize=9, color=INK)
    for ax, cam, col in ((axes[1], "top", BLUE), (axes[2], "side", ORANGE)):
        ax.plot(out[cam][0] - t0, out[cam][1], color=col, lw=1.6)
        ax.set_ylabel(f"sensor moved in\n{cam} image (px)", fontsize=9, color=INK)
    axes[2].set_xlabel("time (s)", fontsize=9, color=INK)
    for ax in axes:
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_title(f"{name}: does the ultrasonic sensor move with the wrist?", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    figp = C.REPO_ROOT / "docs" / "figures" / "real2sim" / f"{name}_sensor_track.png"
    fig.savefig(figp, facecolor=SURF)
    print("wrote", figp)


if __name__ == "__main__":
    main()
