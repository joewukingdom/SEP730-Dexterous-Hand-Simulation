"""Pair every ultrasonic sample with the nearest top and side camera frame (by time).

    python real2sim/recording/align.py run01
    python real2sim/recording/align.py run01 --tol-ms 25 --lat-top-ms 60 --lat-side-ms 60 --lat-us-ms 20

Writes real2sim/data/recordings/<name>/aligned.csv, one row per ultrasonic sample:

  t            ultrasonic sample time (s, corrected for --lat-us-ms)
  us_mm        distance in mm (empty = no echo)
  top_file     nearest top frame (path relative to the recording folder)
  top_dt_ms    top frame time - t (ms); |value| <= tol-ms, else the row is flagged
  side_file    nearest side frame
  side_dt_ms   side frame time - t
  ok           1 if both frames are within tol-ms

Latencies: every stream is stamped on arrival, so its true capture time is
arrival - latency. Pass the measured latencies (see the sync test in the README) so the
pairing compares capture times; with the defaults (0) the pairing is by arrival time.
Rows are never dropped, only flagged, so the table keeps one row per sensor sample.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cameras"))
import camlib as C  # noqa: E402


def load_frames(path: Path, latency_s: float) -> tuple[list[float], list[str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [float(r["t"]) - latency_s for r in rows], [r["file"] for r in rows]


def nearest(ts: list[float], t: float) -> int:
    k = bisect.bisect_left(ts, t)
    if k == 0:
        return 0
    if k == len(ts):
        return len(ts) - 1
    return k if ts[k] - t < t - ts[k - 1] else k - 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--tol-ms", type=float, default=25.0, help="max |frame time - sample time| for ok=1")
    lat_file = HERE / "latency.json"
    lat = json.loads(lat_file.read_text()) if lat_file.exists() else {}
    ap.add_argument("--lat-top-ms", type=float, default=lat.get("lat_top_ms", 0.0),
                    help="default from latency.json (measured with latency.py --event)")
    ap.add_argument("--lat-side-ms", type=float, default=lat.get("lat_side_ms", 0.0))
    ap.add_argument("--lat-us-ms", type=float, default=lat.get("lat_us_ms", 0.0))
    args = ap.parse_args()
    print(f"latencies (ms): ultrasonic {args.lat_us_ms:+.0f}, top {args.lat_top_ms:+.0f}, side {args.lat_side_ms:+.0f}")

    rec = Path(args.name) if Path(args.name).exists() else C.DATA_DIR / "recordings" / args.name
    top_t, top_f = load_frames(rec / "top_frames.csv", args.lat_top_ms / 1000)
    side_t, side_f = load_frames(rec / "side_frames.csv", args.lat_side_ms / 1000)
    with open(rec / "ultrasonic.csv", newline="") as f:
        us = list(csv.DictReader(f))
    gaps = [float(b["t"]) - float(a["t"]) for a, b in zip(us, us[1:])]
    bunched = sum(g < 0.020 for g in gaps)
    if bunched:
        print(f"WARNING: {bunched} ultrasonic samples arrived < 20 ms after the previous one "
              "(the firmware sends one every ~100 ms): their arrival times are not their measurement times")

    n_ok, dts = 0, []
    with open(rec / "aligned.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "us_mm", "top_file", "top_dt_ms", "side_file", "side_dt_ms", "ok"])
        for r in us:
            t = float(r["t"]) - args.lat_us_ms / 1000
            i, j = nearest(top_t, t), nearest(side_t, t)
            dt_top, dt_side = (top_t[i] - t) * 1000, (side_t[j] - t) * 1000
            ok = abs(dt_top) <= args.tol_ms and abs(dt_side) <= args.tol_ms
            n_ok += ok
            dts += [abs(dt_top), abs(dt_side)]
            w.writerow([f"{t:.6f}", r["dist_mm"], top_f[i], f"{dt_top:.1f}", side_f[j], f"{dt_side:.1f}", int(ok)])
    dts.sort()
    print(f"{rec / 'aligned.csv'}: {len(us)} ultrasonic samples, {n_ok} with both frames within "
          f"{args.tol_ms:.0f} ms; |dt| median {dts[len(dts) // 2]:.1f} ms, max {dts[-1]:.1f} ms")
    for name, ts in (("top", top_t), ("side", side_t)):
        gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
        print(f"  {name}: {len(ts)} frames, {len(ts) / (ts[-1] - ts[0]):.1f} fps, "
              f"largest gap {gaps[-1] * 1000:.0f} ms")


if __name__ == "__main__":
    main()
