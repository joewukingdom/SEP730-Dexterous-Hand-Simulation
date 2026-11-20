"""Measure the delay between the streams of a sync recording (cube pushed back and forth
along the ultrasonic beam a few times).

    python real2sim/recording/latency.py sync01

For every top / side frame the cube pose comes from the AprilTags (each camera on its own),
and is turned into "distance from the sensor along its beam". That signal is compared with
the ultrasonic distance at every lag in +-400 ms; the lag with the highest correlation is
how much later that camera's frames arrive than the ultrasonic readings, for the same
physical moment. Prints the align.py flags and writes docs/figures/real2sim/sync_latency.png.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cameras"))
sys.path.insert(0, str(HERE.parent / "cube"))
import camlib as C  # noqa: E402
import cube_tags as CT  # noqa: E402

BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def camera_signal(rec: Path, cam: str, layout: dict, cal: dict, us_cfg: dict):
    intr = C.Intrinsics.load(C.intrinsics_path(cam))
    T_wc = np.asarray(cal["cameras"][cam]["T_world_cam_opencv"])
    pos, beam = np.asarray(us_cfg["pos_m"], float), np.asarray(us_cfg["dir"], float)
    beam /= np.linalg.norm(beam)
    det = CT.detector()
    ts, ds = [], []
    with open(rec / f"{cam}_frames.csv", newline="") as f:
        for r in csv.DictReader(f):
            T_cc, used, err = CT.cube_pose_joint(cv2.imread(str(rec / r["file"])), intr, layout, det)
            if T_cc is None:
                continue
            c = (T_wc @ T_cc)[:3, 3]
            ts.append(float(r["t"]))
            ds.append(((c - pos) @ beam - CT.HALF) * 1000)  # near face, mm
    return np.array(ts), np.array(ds)


def best_lag(t_ref, x_ref, t_sig, x_sig, lags_ms):
    """Lag (ms) maximising corr(x_ref(t), x_sig(t + lag))."""
    out = []
    for lag in lags_ms:
        tq = t_ref + lag / 1000
        ok = (tq >= t_sig[0]) & (tq <= t_sig[-1])
        if ok.sum() < 20:
            out.append(np.nan)
            continue
        y = np.interp(tq[ok], t_sig, x_sig)
        out.append(np.corrcoef(x_ref[ok], y)[0, 1])
    out = np.array(out)
    return lags_ms[int(np.nanargmax(out))], out


def main() -> None:
    rec = C.DATA_DIR / "recordings" / sys.argv[1]
    layout = json.loads(CT.LAYOUT.read_text())
    cal = json.loads((C.CALIB_DIR / "cameras.json").read_text())
    us_cfg = C.load_rig(C.HERE / "rig.yaml")["ultrasonic"]
    with open(rec / "ultrasonic.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["dist_mm"]]
    t_us = np.array([float(r["t"]) for r in rows])
    d_us = np.array([float(r["dist_mm"]) for r in rows])
    lags = np.arange(-400, 401, 1.0)
    sig, res = {}, {}
    for cam in ("top", "side"):
        t, d = camera_signal(rec, cam, layout, cal, us_cfg)
        sig[cam] = (t, d)
        if len(t) < 20:
            print(f"{cam}: only {len(t)} frames with the cube found, skipped")
            continue
        lag, corr = best_lag(t_us, d_us, t, d, lags)
        res[cam] = (lag, corr)
        print(f"{cam}: cube found in {len(t)} frames; best lag {lag:+.0f} ms (corr {np.nanmax(corr):.3f}) "
              f"-> {cam} frames arrive {lag:+.0f} ms {'after' if lag > 0 else 'before'} the ultrasonic reading "
              "of the same moment")
    if "top" in res and "side" in res:
        print(f"align.py flags: --lat-us-ms 0 --lat-top-ms {res['top'][0]:.0f} --lat-side-ms {res['side'][0]:.0f}"
              "   (relative to the ultrasonic stream)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), dpi=200, sharex=True)
    fig.patch.set_facecolor(SURF)
    for ax, shift in zip(axes, (False, True)):
        ax.set_facecolor(SURF)
        ax.plot(t_us - t_us[0], d_us, color=BLUE, lw=2, label="ultrasonic")
        for cam, col in (("top", ORANGE), ("side", AQUA)):
            if cam in res:
                t, d = sig[cam]
                off = res[cam][0] / 1000 if shift else 0.0
                ax.plot(t - off - t_us[0], d, color=col, lw=1.5, label=f"{cam} camera (tags)")
        ax.set_ylabel("distance (mm)", color=INK, fontsize=9)
        ax.set_title("as recorded (arrival times)" if not shift else "after removing the measured lags",
                     loc="left", color=INK, fontsize=10)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].set_xlabel("time (s)", color=INK, fontsize=9)
    fig.tight_layout()
    out = C.REPO_ROOT / "docs" / "figures" / "real2sim" / "sync_latency.png"
    fig.savefig(out, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
