"""Turn a short synchronized recording into a pose-check folder (top.png, side.png,
real_reading.json) using frames taken at the same moment.

    python real2sim/recording/record.py tagcheck2 --seconds 3
    python real2sim/cube/snapshot_from_recording.py tagcheck2
    python real2sim/cube/check_pose.py real2sim/data/recordings/tagcheck2/snapshot
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cameras"))
import camlib as C  # noqa: E402


def main() -> None:
    rec = C.DATA_DIR / "recordings" / sys.argv[1]
    frames = {}
    for cam in ("top", "side"):
        with open(rec / f"{cam}_frames.csv", newline="") as f:
            frames[cam] = [(float(r["t"]), r["file"]) for r in csv.DictReader(f)]
    t_mid = frames["top"][len(frames["top"]) // 2][0]
    out = rec / "snapshot"
    out.mkdir(exist_ok=True)
    picked = {}
    for cam, rows in frames.items():
        t, fname = min(rows, key=lambda r: abs(r[0] - t_mid))
        cv2.imwrite(str(out / f"{cam}.png"), cv2.imread(str(rec / fname)))
        picked[cam] = (fname, round((t - t_mid) * 1000, 1))
    with open(rec / "ultrasonic.csv", newline="") as f:
        vals = [float(r["dist_mm"]) for r in csv.DictReader(f) if r["dist_mm"]]
    (out / "real_reading.json").write_text(json.dumps({"median_mm": statistics.median(vals) if vals else None,
                                                       "n": len(vals), "frames": picked}, indent=2))
    print(f"snapshot at t={t_mid:.3f}s: {picked}; ultrasonic median "
          f"{statistics.median(vals) if vals else 'none'} mm over {len(vals)} readings")


if __name__ == "__main__":
    main()
