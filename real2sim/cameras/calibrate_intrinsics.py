"""Intrinsic calibration (K + distortion) of one camera from ChArUco images.

    python real2sim/cameras/calibrate_intrinsics.py intrinsics_top --cam top

Reads real2sim/data/<session>/<cam>/*.png, writes real2sim/cameras/calib/<cam>_intrinsics.json.
Images where the board is barely visible are skipped; after a first solve, views whose
reprojection error is far above the rest are dropped and the solve is repeated.

What good looks like: RMS < 0.5 px, 20+ images, and the coverage map showing corners
in all cells (board seen near every image edge and corner, tilted up to ~45 deg).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import camlib as C


def collect(images: list[Path], det: C.BoardDetector, min_corners: int):
    obj, img, used, size = [], [], [], None
    for p in images:
        im = cv2.imread(str(p))
        if im is None:
            continue
        size = (im.shape[1], im.shape[0])
        corners, ids = det.detect(im)
        if len(ids) < min_corners:
            continue
        obj.append(det.object_points(ids).astype(np.float32))
        img.append(corners.reshape(-1, 1, 2))
        used.append(p)
    return obj, img, used, size


def calibrate(obj, img, size, flags: int = 0):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
    per_view = []
    for o, i, rv, tv in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj - i) ** 2, axis=2)))))
    return rms, K, dist, np.array(per_view)


def coverage(img, size, grid=(4, 3)) -> np.ndarray:
    W, H = size
    cov = np.zeros(grid[::-1], int)
    for pts in img:
        for x, y in pts.reshape(-1, 2):
            cov[min(int(y / H * grid[1]), grid[1] - 1), min(int(x / W * grid[0]), grid[0] - 1)] += 1
    return cov


def run(image_dir: Path, board_cfg: dict, *, min_corners: int = 8, zero_tangent: bool = False,
        verbose: bool = True) -> C.Intrinsics:
    det = C.BoardDetector(board_cfg)
    images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    obj, img, used, size = collect(images, det, min_corners)
    if len(obj) < 6:
        raise SystemExit(f"only {len(obj)} usable images in {image_dir} (need >= 6, ideally 20+)")
    flags = cv2.CALIB_ZERO_TANGENT_DIST if zero_tangent else 0
    rms, K, dist, per_view = calibrate(obj, img, size, flags)
    keep = per_view < max(3 * np.median(per_view), 0.5)
    if not keep.all() and keep.sum() >= 6:
        if verbose:
            print("dropping outlier views:", [used[i].name for i in np.flatnonzero(~keep)])
        obj = [o for o, k in zip(obj, keep) if k]
        img = [i for i, k in zip(img, keep) if k]
        rms, K, dist, per_view = calibrate(obj, img, size, flags)
    if verbose:
        W, H = size
        print(f"images used {len(obj)}/{len(images)}   RMS {rms:.3f} px   size {W}x{H}")
        print(f"fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}   "
              f"vertical FOV {np.degrees(2 * np.arctan(H / 2 / K[1, 1])):.1f} deg")
        print("dist", np.round(dist.ravel(), 4))
        print("corner coverage (columns = image left->right):\n", coverage(img, size))
        if rms > 1.0:
            print("WARNING: RMS > 1 px -- check the board is flat and square_mm is the measured value")
    return C.Intrinsics(size[0], size[1], K.tolist(), dist.ravel().tolist(), float(rms), len(obj))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="folder under real2sim/data/ (or a path)")
    ap.add_argument("--cam", required=True)
    ap.add_argument("--rig", default=C.HERE / "rig.yaml")
    ap.add_argument("--min-corners", type=int, default=8)
    args = ap.parse_args()
    session = Path(args.session)
    if not session.exists():
        session = C.DATA_DIR / args.session
    intr = run(session / args.cam, C.load_rig(args.rig)["board"], min_corners=args.min_corners)
    out = C.intrinsics_path(args.cam)
    intr.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
