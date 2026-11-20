"""Correct a calibrated camera that was bumped slightly after its extrinsic capture.

    python real2sim/cameras/correct_bump.py top real2sim/data/extr_top/top/008.png real2sim/data/extr_top/after_bump.png

`before` is a frame from the extrinsic capture, `after` a frame taken now. The static
background of the two frames is matched; a bump of a clamped camera is (almost) a pure
rotation, so the matched rays give R (old camera coords -> new camera coords). The camera's
pose in calib/cameras.json is rotated accordingly and the _real2sim scene is rewritten.
The residual tells whether the pure-rotation assumption holds (~1 px: yes; several px: the
camera also moved, redo the extrinsic capture instead).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import calibrate_extrinsics as CE
import camlib as C


def bump_rotation(before: np.ndarray, after: np.ndarray, intr: C.Intrinsics) -> tuple[np.ndarray, dict]:
    K, dist = intr.K_np, intr.dist_np
    a = cv2.cvtColor(cv2.undistort(before, K, dist), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.undistort(after, K, dist), cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(a, None)
    k2, d2 = orb.detectAndCompute(b, None)
    ms = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    p1 = np.float32([k1[m.queryIdx].pt for m in ms])
    p2 = np.float32([k2[m.trainIdx].pt for m in ms])
    _, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
    inl = inl.ravel().astype(bool)
    rays = []
    for p in (p1[inl], p2[inl]):
        n = cv2.undistortPoints(p.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        v = np.c_[n, np.ones(len(n))]
        rays.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    v1, v2 = rays
    U, _, Vt = np.linalg.svd(v2.T @ v1)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt  # v2 ~ R v1
    ang = np.arccos(np.clip(np.sum((v1 @ R.T) * v2, axis=1), -1, 1))
    stats = {"deg": C.rot_angle_deg(R, np.eye(3)), "inliers": int(inl.sum()),
             "residual_median_px": float(np.median(ang) * K[0, 0])}
    return R, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cam")
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--rig", default=C.HERE / "rig.yaml")
    args = ap.parse_args()

    intr = C.Intrinsics.load(C.intrinsics_path(args.cam))
    R, stats = bump_rotation(cv2.imread(str(args.before)), cv2.imread(str(args.after)), intr)
    print(f"{args.cam}: bump {stats['deg']:.3f} deg from {stats['inliers']} background matches, "
          f"residual {stats['residual_median_px']:.2f} px")
    if stats["residual_median_px"] > 2.0:
        raise SystemExit("residual too large for a pure rotation: redo the extrinsic capture instead")

    cal_path = C.CALIB_DIR / "cameras.json"
    cal = json.loads(cal_path.read_text())
    cam = cal["cameras"][args.cam]
    T = np.asarray(cam["T_world_cam_opencv"]) @ C.make_T(R.T, [0, 0, 0])
    R_mj = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    cam.update({"T_world_cam_opencv": T.tolist(), "pos_m": T[:3, 3].tolist(), "look_dir_world": T[:3, 2].tolist(),
                "mujoco_xyaxes": np.concatenate([R_mj[:, 0], R_mj[:, 1]]).tolist()})
    cam.setdefault("bump_corrections", []).append(
        {**stats, "R_new_old": R.tolist(), "before": str(args.before), "after": str(args.after)})
    cal_path.write_text(json.dumps(cal, indent=2))
    scene = CE.write_scene(C.load_rig(args.rig), cal)
    print(f"updated {cal_path} and {scene}")


if __name__ == "__main__":
    main()
