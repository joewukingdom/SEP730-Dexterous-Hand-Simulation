"""End-to-end check of the camera real2sim pipeline on synthetic "real" images.

MuJoCo plays the real rig: two cameras with known intrinsics, lens distortion and poses
photograph the ChArUco board. Like Joe's bench, the top camera looks down at a board lying
flat on the bench, and the side camera, almost level with the bench, looks at a board
standing up; each camera is calibrated from its own session and board placement. The pipeline (calibrate_intrinsics -> calibrate_extrinsics
-> overlay) must recover the cameras from the images alone, and the generated MuJoCo
scene must reproduce the "real" images, also for a held-out board placement.

    python -m pytest tests/test_real2sim_cameras.py -s      # -s prints the numbers
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):
    pytest.skip("needs opencv-contrib-python (cv2.aruco)", allow_module_level=True)
pytest.importorskip("yaml")
import mujoco  # noqa: E402

CAM_DIR = Path(__file__).resolve().parents[1] / "real2sim" / "cameras"
sys.path.insert(0, str(CAM_DIR))
import calibrate_extrinsics  # noqa: E402
import calibrate_intrinsics  # noqa: E402
import camlib as C  # noqa: E402
import overlay  # noqa: E402

W, H = 1280, 720
GT = {  # name: (eye, target, fx, fy, cx, cy, dist)
    "top": ((0.10, -0.10, 0.62), (0.10, -0.12, 0.0), 1000.0, 1000.0, 650.0, 355.0,
            [-0.08, 0.05, 0.0, 0.0, 0.0]),
    "side": ((0.12, 0.62, 0.20), (0.10, 0.05, 0.13), 980.0, 985.0, 628.0, 368.0,
             [-0.06, 0.03, 0.0, 0.0, 0.0]),
}
BOARD = {  # per-camera calibration placement
    "top": {"pos_m": [0.10, -0.20, 0.003], "yaw_deg": 0.0, "tilt_deg": 0.0},
    "side": {"pos_m": [0.10, 0.15, 0.0875], "yaw_deg": 0.0, "tilt_deg": -90.0},
}
HELD_OUT_BOARD = {
    "top": {"pos_m": [0.14, -0.17, 0.003], "yaw_deg": 25.0, "tilt_deg": 0.0},
    "side": {"pos_m": [0.06, 0.19, 0.0875], "yaw_deg": 20.0, "tilt_deg": -90.0},
}


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return C.make_T(np.column_stack([x, np.cross(z, x), z]), eye)


def distort(img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Pinhole render -> what a camera with this distortion would see."""
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([xs, ys], -1).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-6)
    if hasattr(cv2, "undistortPointsIter"):  # OpenCV 4.x
        und = cv2.undistortPointsIter(pts, K, dist, None, K, crit)
    else:  # OpenCV 5.x folded the criteria into undistortPoints
        und = cv2.undistortPoints(pts, K, dist, None, None, K, crit)
    m = und.reshape(h, w, 2)
    return cv2.remap(img, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderValue=(0, 0, 0))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("real2sim")
    rig = C.load_rig(CAM_DIR / "rig.yaml")
    rig.pop("bench", None)  # synthetic world: boards sit near z = 0, below the real bench height
    rig["cameras"] ={n: {"device": i, "width": W, "height": H, "board_in_world": BOARD[n]}
                      for i, n in enumerate(GT)}
    gt_cams = {n: C.MjCamera(n, look_at(e, t), W, H, fx, fy, cx, cy) for n, (e, t, fx, fy, cx, cy, _) in GT.items()}
    gt_dist = {n: np.asarray(v[-1]) for n, v in GT.items()}

    old_calib = C.CALIB_DIR
    C.CALIB_DIR = tmp / "calib"
    created = [C.SCENES_DIR / n for n in ("_test_gt.xml", "_test_gt_board.png", "_test_real2sim.xml", "_test_board.png")]
    try:
        C.write_board_texture(rig["board"], "_test_gt_board.png")
        T_wb = C.T_world_boardcv(rig["board"], rig["board_in_world"])
        (C.SCENES_DIR / "_test_gt.xml").write_text(C.build_scene_xml(
            rig["scene"], rig["board"], T_wb, list(gt_cams.values()), board_mocap=True, texture_file="_test_gt_board.png"))
        m = mujoco.MjModel.from_xml_path(str(C.SCENES_DIR / "_test_gt.xml"))
        d = mujoco.MjData(m)

        def shoot(session: str, name: str, idx: int, T_world_board: np.ndarray) -> None:
            overlay.set_board_pose(m, d, T_world_board)
            img = C.render_camera(m, d, name)
            out = tmp / session / name
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{idx:03d}.png"), distort(img, gt_cams[name].K, gt_dist[name]))

        # intrinsics: board held in front of each camera at varied distance / offset / tilt
        rng = np.random.default_rng(0)
        lx, ly = C.board_size_m(rig["board"])
        for name, cam in gt_cams.items():
            for i in range(18):
                tilt = C.rot_x(math.radians(rng.uniform(-35, 35))) @ \
                    np.array(C.make_T(C.rot_z(0), [0, 0, 0]))[:3, :3]
                ry = math.radians(rng.uniform(-35, 35))
                tilt = tilt @ np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
                z = rng.uniform(0.32, 0.55)
                off = np.array([rng.uniform(-0.35, 0.35) * z, rng.uniform(-0.2, 0.2) * z, z])
                T_cam_board = C.make_T(tilt, off) @ C.make_T(np.eye(3), [-lx / 2, -ly / 2, 0])
                shoot("intr", name, i, cam.T_world_camcv @ T_cam_board)
        for name in gt_cams:  # each camera on its own, with its own board placement
            for i in range(3):
                shoot(f"extr_{name}", name, i, C.T_world_boardcv(rig["board"], BOARD[name]))
            shoot(f"heldout_{name}", name, 0, C.T_world_boardcv(rig["board"], HELD_OUT_BOARD[name]))
        yield {"tmp": tmp, "rig": rig, "gt": gt_cams, "gt_dist": gt_dist}
    finally:
        C.CALIB_DIR = old_calib
        for p in created:
            p.unlink(missing_ok=True)


def test_intrinsics_recovered(world):
    for name, cam in world["gt"].items():
        intr = calibrate_intrinsics.run(world["tmp"] / "intr" / name, world["rig"]["board"], verbose=False)
        intr.save(C.intrinsics_path(name))
        K = intr.K_np
        print(f"\n{name}: rms {intr.rms_px:.3f}px  fx {K[0, 0]:.1f}/{cam.fx}  fy {K[1, 1]:.1f}/{cam.fy}  "
              f"cx {K[0, 2]:.1f}/{cam.cx}  cy {K[1, 2]:.1f}/{cam.cy}  k1 {intr.dist[0]:.3f}/{world['gt_dist'][name][0]}")
        assert intr.rms_px < 0.5
        assert abs(K[0, 0] / cam.fx - 1) < 0.01 and abs(K[1, 1] / cam.fy - 1) < 0.01
        assert abs(K[0, 2] - cam.cx) < 4 and abs(K[1, 2] - cam.cy) < 4
        assert abs(intr.dist[0] - world["gt_dist"][name][0]) < 0.03


def test_extrinsics_and_overlay(world):
    sessions = {n: world["tmp"] / f"extr_{n}" for n in world["gt"]}
    out = calibrate_extrinsics.run(None, world["rig"], cam_sessions=sessions, scene_name="_test_real2sim.xml",
                                   texture_name="_test_board.png", verbose=False)
    for name, cam in world["gt"].items():
        T = np.asarray(out["cameras"][name]["T_world_cam_opencv"])
        dp = np.linalg.norm(T[:3, 3] - cam.T_world_camcv[:3, 3]) * 1000
        dr = C.rot_angle_deg(T[:3, :3], cam.T_world_camcv[:3, :3])
        print(f"\n{name}: camera position error {dp:.2f} mm, rotation error {dr:.3f} deg")
        assert dp < 5.0 and dr < 0.3

    scene = Path(out["scene_file"])
    checks = [("calib", None, None, None)] + [
        ("heldout", world["tmp"] / f"heldout_{n}", HELD_OUT_BOARD[n], [n]) for n in world["gt"]]
    for label, session, biw, cams in checks:
        res = overlay.run(session, scene_file=scene, board_in_world=biw, cams=cams, show=False, save=False,
                          verbose=False)
        assert res
        for name, r in res.items():
            print(f"{label} {name}: real-vs-sim corners {r['n']}, mean {r['mean_px']:.2f} px "
                  f"({r['mean_mm']:.2f} mm), max {r['max_px']:.2f} px")
            assert r["n"] >= 12
            assert r["mean_px"] < 0.6 and r["mean_mm"] < 0.5
