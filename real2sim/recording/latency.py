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


def movement_onsets(t: np.ndarray, x: np.ndarray, jump_mm: float, settle_s: float = 0.5) -> np.ndarray:
    """Times where the signal leaves a stable level: first sample more than jump_mm away from
    the median of the preceding settle_s window, at least settle_s after the previous onset."""
    out, last = [], -np.inf
    for i in range(len(t)):
        if t[i] - last < settle_s:
            continue
        w = (t >= t[i] - settle_s) & (t < t[i])
        if w.sum() < 3:
            continue
        if abs(x[i] - np.median(x[w])) > jump_mm and np.ptp(x[w]) < jump_mm:
            out.append(t[i])
            last = t[i]
    return np.array(out)


def onset_lag(ref: np.ndarray, sig: np.ndarray, max_s: float = 0.4) -> tuple[float, int, float]:
    """Median of (sig onset - nearest ref onset) over onsets matched within max_s."""
    d = [s - ref[np.argmin(abs(ref - s))] for s in sig if len(ref) and abs(ref - s).min() < max_s]
    if not d:
        return float("nan"), 0, float("nan")
    d = np.array(d) * 1000
    return float(np.median(d)), len(d), float(np.median(abs(d - np.median(d))))


def beam_roi(cam: str, cal: dict, us_cfg: dict, length_m: float = 0.08, margin_px: int = 60):
    """Pixel box around the first `length_m` of the ultrasonic beam in a camera image."""
    T_wc = np.asarray(cal["cameras"][cam]["T_world_cam_opencv"])
    intr = C.Intrinsics.load(C.intrinsics_path(cam))
    pos, beam = np.asarray(us_cfg["pos_m"], float), np.asarray(us_cfg["dir"], float)
    beam /= np.linalg.norm(beam)
    pts = np.array([pos + s * beam + off for s in np.linspace(0.01, length_m, 8)
                    for off in ([0, 0, 0.015], [0, 0, -0.015], [0, 0.015, 0], [0, -0.015, 0])])
    T_cw = C.inv_T(T_wc)
    rv, _ = cv2.Rodrigues(T_cw[:3, :3])
    uv, _ = cv2.projectPoints(pts, rv, T_cw[:3, 3], intr.K_np, intr.dist_np)
    uv = uv.reshape(-1, 2)
    x0, y0 = np.floor(uv.min(0)).astype(int) - margin_px
    x1, y1 = np.ceil(uv.max(0)).astype(int) + margin_px
    return max(x0, 0), max(y0, 0), min(x1, 1920), min(y1, 1080)


def roi_change(rec: Path, cam: str, roi) -> tuple[np.ndarray, np.ndarray]:
    """Mean absolute grey-level change inside the beam ROI between consecutive frames."""
    x0, y0, x1, y1 = roi
    ts, ch, prev = [], [], None
    with open(rec / f"{cam}_frames.csv", newline="") as f:
        for r in csv.DictReader(f):
            g = cv2.cvtColor(cv2.imread(str(rec / r["file"]))[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                ts.append(float(r["t"]))
                ch.append(float(np.abs(g - prev).mean()))
            prev = g
    return np.array(ts), np.array(ch)


def event_onsets(t: np.ndarray, x: np.ndarray, thr: float, quiet_s: float = 0.4) -> np.ndarray:
    """Times where x first exceeds thr after at least quiet_s below it."""
    out, last_active = [], -np.inf
    for ti, xi in zip(t, x):
        if xi > thr:
            if ti - last_active > quiet_s:
                out.append(ti)
            last_active = ti
    return np.array(out)


def event_mode(rec: Path, cal: dict, us_cfg: dict) -> None:
    """Sharp-event sync test: cube removed, a card / hand pushed into the beam and pulled out."""
    with open(rec / "ultrasonic.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    t_us = np.array([float(r["t"]) for r in rows])
    near = np.array([1.0 if (r["dist_mm"] and float(r["dist_mm"]) <= 150) else 0.0 for r in rows])
    # an ultrasonic event: the first reading of a change between "nothing near" and "something near"
    us_on = np.array([t_us[i] for i in range(1, len(near)) if near[i] != near[i - 1]])
    print(f"ultrasonic: {len(rows)} readings, {len(us_on)} in/out transitions")
    on = {"ultrasonic": us_on}
    for cam in ("top", "side"):
        roi = beam_roi(cam, cal, us_cfg)
        t, ch = roi_change(rec, cam, roi)
        thr = max(np.median(ch) * 6, np.percentile(ch, 50) + 5 * np.median(abs(ch - np.median(ch))) + 1)
        on[cam] = event_onsets(t, ch, thr, quiet_s=0.25)
        print(f"{cam}: ROI {roi}, change threshold {thr:.1f}, {len(on[cam])} events")
    for a, b in (("ultrasonic", "top"), ("ultrasonic", "side"), ("top", "side")):
        lag, n, mad = onset_lag(on[a], on[b], max_s=0.35)
        print(f"  {b} event - {a} event: median {lag:+.0f} ms over {n} events (MAD {mad:.0f} ms)")
    print("note: an ultrasonic event is stamped at the first reading after the change; with one reading every "
          "~100 ms that is on average ~50 ms after the physical moment (included in the numbers above).")


def main() -> None:
    if "--event" in sys.argv:
        sys.argv.remove("--event")
        rec = C.DATA_DIR / "recordings" / sys.argv[1]
        event_mode(rec, json.loads((C.CALIB_DIR / "cameras.json").read_text()),
                   C.load_rig(C.HERE / "rig.yaml")["ultrasonic"])
        return
    rec = C.DATA_DIR / "recordings" / sys.argv[1]
    layout = json.loads(CT.LAYOUT.read_text())
    cal = json.loads((C.CALIB_DIR / "cameras.json").read_text())
    us_cfg = C.load_rig(C.HERE / "rig.yaml")["ultrasonic"]
    with open(rec / "ultrasonic.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["dist_mm"]]
    t_us = np.array([float(r["t"]) for r in rows])
    d_us = np.array([float(r["dist_mm"]) for r in rows])
    # drop echoes from elsewhere (dead zone / tilted cube send the sound away: 100-1500 mm
    # readings) and single-sample spikes (3-point median), so they do not drive the correlation
    med = np.array([np.median(d_us[max(0, i - 1):i + 2]) for i in range(len(d_us))])
    keep = (med >= 15) & (med <= 150)
    print(f"ultrasonic: {len(d_us)} readings, {int((~keep).sum())} outside 15-150 mm after a 3-point median, dropped")
    t_us, d_us = t_us[keep], med[keep]
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
    for cam, (lag, corr) in res.items():
        ok = lags[np.nan_to_num(corr, nan=-1) >= np.nanmax(corr) - 0.01]
        print(f"  {cam}: lags within 0.01 of the best correlation span {ok.min():+.0f} .. {ok.max():+.0f} ms")
    if "top" in res and "side" in res:
        # the two cameras directly (both 30 fps): far finer than against the 10 Hz ultrasonic
        (tt, dt_), (ts, ds) = sig["top"], sig["side"]
        lag_ts, corr_ts = best_lag(tt, dt_, ts, ds, lags)
        ok = lags[np.nan_to_num(corr_ts, nan=-1) >= np.nanmax(corr_ts) - 0.01]
        print(f"side vs top directly: side frames arrive {lag_ts:+.0f} ms relative to top "
              f"(corr {np.nanmax(corr_ts):.3f}, span {ok.min():+.0f} .. {ok.max():+.0f} ms); "
              f"via the ultrasonic: {res['side'][0] - res['top'][0]:+.0f} ms")
        print(f"align.py flags: --lat-us-ms 0 --lat-top-ms {res['top'][0]:.0f} --lat-side-ms {res['side'][0]:.0f}"
              "   (relative to the ultrasonic stream)")

    # onsets of the pushes: the most direct timing evidence (the ultrasonic samples every
    # ~100 ms, so its onset comes on average ~50 ms after the physical start)
    on = {"ultrasonic": movement_onsets(t_us, d_us, 4.0)}
    for cam in res:
        on[cam] = movement_onsets(*sig[cam], 4.0)
    print("onsets found: " + ", ".join(f"{k} {len(v)}" for k, v in on.items()))
    for a, b in (("ultrasonic", "top"), ("ultrasonic", "side"), ("top", "side")):
        if a in on and b in on:
            lag, n, mad = onset_lag(on[a], on[b])
            print(f"  {b} onset - {a} onset: median {lag:+.0f} ms over {n} pushes (MAD {mad:.0f} ms)")

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
