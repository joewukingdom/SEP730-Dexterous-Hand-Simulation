"""Did the forearm tower (the reference of the camera extrinsics and the ultrasonic pose) move
during a recording? Compares the first and the last frame of each camera inside the tower's
image region (projected from the sim tower box) by ORB feature matching.

    python real2sim/recording/tower_check.py run03
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
import camlib as C  # noqa: E402

TOWER = np.array([[x, y, z] for x in (-0.058, 0.099) for y in (-0.057, 0.037) for z in (0.112, 0.208)])


def roi(cam: str, cal: dict, margin: int = 20):
    T = np.asarray(cal["cameras"][cam]["T_world_cam_opencv"])
    intr = C.Intrinsics.load(C.intrinsics_path(cam))
    Tcw = C.inv_T(T)
    uv, _ = cv2.projectPoints(TOWER, cv2.Rodrigues(Tcw[:3, :3])[0], Tcw[:3, 3], intr.K_np, intr.dist_np)
    uv = uv.reshape(-1, 2)
    x0, y0 = np.maximum(np.floor(uv.min(0)).astype(int) - margin, 0)
    x1, y1 = np.minimum(np.ceil(uv.max(0)).astype(int) + margin, [1920, 1080])
    return x0, y0, x1, y1


def main() -> None:
    rec = C.DATA_DIR / "recordings" / sys.argv[1]
    cal = json.loads((C.CALIB_DIR / "cameras.json").read_text())
    for cam in ("top", "side"):
        files = [r["file"] for r in csv.DictReader(open(rec / f"{cam}_frames.csv", newline=""))]
        x0, y0, x1, y1 = roi(cam, cal)
        a = cv2.imread(str(rec / files[0]), cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1]
        b = cv2.imread(str(rec / files[-1]), cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1]
        orb = cv2.ORB_create(3000)
        k1, d1 = orb.detectAndCompute(a, None)
        k2, d2 = orb.detectAndCompute(b, None)
        ms = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
        p1 = np.float32([k1[m.queryIdx].pt for m in ms])
        p2 = np.float32([k2[m.trainIdx].pt for m in ms])
        _, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        inl = inl.ravel().astype(bool)
        dlt = np.linalg.norm(p2[inl] - p1[inl], axis=1)
        print(f"{cam}: tower region {x1 - x0}x{y1 - y0} px, {inl.sum()} matched features, "
              f"shift first->last frame median {np.median(dlt):.2f} px, 90% {np.percentile(dlt, 90):.2f} px")


if __name__ == "__main__":
    main()
