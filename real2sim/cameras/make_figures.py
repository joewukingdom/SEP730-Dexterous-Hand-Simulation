"""Figures for docs/real2sim_camera_calibration.md (needs the captured data in real2sim/data/).

    python real2sim/cameras/make_figures.py

Writes into docs/figures/real2sim/:
  coverage_top.jpg / coverage_side.jpg   every detected corner of the intrinsic sessions
  final_overlay.jpg                      calibrated sim vs undistorted real, both cameras
                                         (shadows off, so only the geometry is compared)
`final_overlay` uses real2sim/data/live/<cam>.png (a frame of the empty bench, board removed).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

import camlib as C

OUT = C.REPO_ROOT / "docs" / "figures" / "real2sim"


def coverage(session: str, cam: str) -> np.ndarray:
    rig = C.load_rig(C.HERE / "rig.yaml")
    det = C.BoardDetector(rig["board"])
    files = sorted((C.DATA_DIR / session / cam).glob("*.png"))
    base = (cv2.imread(str(files[0])) * 0.35).astype(np.uint8)
    n_img = n_pts = 0
    for f in files:
        corners, ids = det.detect(cv2.imread(str(f)))
        if len(ids) < 8:
            continue
        n_img += 1
        n_pts += len(ids)
        for x, y in corners:
            cv2.circle(base, (int(x), int(y)), 7, (0, 230, 255), -1)
    for x in range(0, 1920, 480):
        cv2.line(base, (x, 0), (x, 1079), (255, 255, 255), 2)
    for y in (360, 720):
        cv2.line(base, (0, y), (1919, y), (255, 255, 255), 2)
    cv2.putText(base, f"{cam}: {n_pts} corners from {n_img} images", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (255, 255, 255), 5)
    return base


def final_overlay() -> np.ndarray:
    rig = C.load_rig(C.HERE / "rig.yaml")
    scene = (C.SCENES_DIR / rig["scene"].replace(".xml", "_real2sim.xml")).resolve()
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 1080, 1920)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    rows = []
    for cam in ("top", "side"):
        intr = C.Intrinsics.load(C.intrinsics_path(cam))
        und = cv2.undistort(cv2.imread(str(C.DATA_DIR / "live" / f"{cam}.png")), intr.K_np, intr.dist_np)
        r.update_scene(d, camera=cam, scene_option=opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        sim = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        edges = cv2.dilate(cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY), 30, 90), np.ones((3, 3), np.uint8))
        ov = und.copy()
        ov[edges > 0] = (255, 0, 255)
        tiles = [und, ov, cv2.addWeighted(und, 0.5, sim, 0.5, 0)]
        for t, label in zip(tiles, ("real (undistorted)", "sim edges on real", "50/50 blend")):
            cv2.putText(t, f"{cam}: {label}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 9)
            cv2.putText(t, f"{cam}: {label}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)
        rows.append(np.hstack([cv2.resize(t, (960, 540)) for t in tiles]))
    return np.vstack(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for session, cam in (("intr_top", "top"), ("intr_side", "side")):
        cv2.imwrite(str(OUT / f"coverage_{cam}.jpg"), cv2.resize(coverage(session, cam), (960, 540)),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(OUT / "final_overlay.jpg"), final_overlay(), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
