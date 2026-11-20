"""Cube pose from the AprilTags in both cameras, compared with each other and drawn over the
real frames with the sim cube; plus the sim ultrasonic reading for that pose.

    python real2sim/cube/check_pose.py real2sim/data/us_check/tagcheck

The folder holds top.png, side.png (and optionally real_reading.json from the sensor).
Writes <folder>/pose_check.jpg and prints the poses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cameras"))
import camlib as C  # noqa: E402
import cube_tags as CT  # noqa: E402


def world_pose(cam: str, img: np.ndarray, layout: dict, cal: dict):
    intr = C.Intrinsics.load(C.intrinsics_path(cam))
    T_cc, used, err = CT.cube_pose_joint(img, intr, layout)
    if T_cc is None:
        return None, used, err
    return np.asarray(cal["cameras"][cam]["T_world_cam_opencv"]) @ T_cc, used, err


def main() -> None:
    folder = Path(sys.argv[1])
    layout = json.loads(CT.LAYOUT.read_text())
    cal = json.loads((C.CALIB_DIR / "cameras.json").read_text())
    rig = C.load_rig(C.HERE / "rig.yaml")
    imgs = {c: cv2.imread(str(folder / f"{c}.png")) for c in ("top", "side")}
    poses = {}
    for cam, img in imgs.items():
        T, used, err = world_pose(cam, img, layout, cal)
        poses[cam] = T
        if T is None:
            print(f"{cam}: no tag found")
            continue
        up = T[:3, 2]  # cube +z = green face (sim red face) normal
        print(f"{cam}: tags {used}, reproj {err:.2f} px, centre {np.round(T[:3, 3], 4)} m, "
              f"green-face normal {np.round(up, 3)} ({np.degrees(np.arccos(np.clip(up[2], -1, 1))):.1f} deg from up)")
    if poses.get("top") is not None and poses.get("side") is not None:
        a, b = poses["top"], poses["side"]
        print(f"top vs side: centre {np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000:.1f} mm apart, "
              f"rotation {C.rot_angle_deg(a[:3, :3], b[:3, :3]):.1f} deg apart")
    views = [(imgs[c], C.Intrinsics.load(C.intrinsics_path(c)), np.asarray(cal["cameras"][c]["T_world_cam_opencv"]))
             for c in ("top", "side")]
    T, info = CT.cube_pose_fused(views, layout)
    if T is None:
        raise SystemExit("cube not found in either camera")
    up = T[:3, 2]
    print(f"FUSED: centre {np.round(T[:3, 3], 4)} m, green-face normal {np.round(up, 3)} "
          f"({np.degrees(np.arccos(np.clip(up[2], -1, 1))):.1f} deg from up), from "
          + ", ".join(f"{('top', 'side')[vi]} tags {ids}" for vi, (ids, e, _) in info.items()))
    for cam in ("top", "side"):
        if poses.get(cam) is not None:
            print(f"  {cam} alone vs both: {np.linalg.norm(poses[cam][:3, 3] - T[:3, 3]) * 1000:.1f} mm, "
                  f"{C.rot_angle_deg(poses[cam][:3, :3], T[:3, :3]):.1f} deg")

    scene = (C.SCENES_DIR / rig["scene"].replace(".xml", "_real2sim.xml")).resolve()
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    j = m.joint("cube_freejoint").qposadr[0]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, T[:3, :3].flatten())
    d.qpos[j:j + 3] = T[:3, 3]
    d.qpos[j + 3:j + 7] = q
    mujoco.mj_forward(m, d)
    sim = C.ultrasonic_reading(m, d, rig["ultrasonic"])
    real = None
    if (folder / "real_reading.json").exists():
        real = json.loads((folder / "real_reading.json").read_text())["median_mm"]
    print(f"ultrasonic: sim {'no echo' if sim is None else f'{sim * 1000:.1f} mm'}, "
          f"real {'-' if real is None else f'{real:.1f} mm'}")

    tiles = []
    for cam, img in imgs.items():
        intr = C.Intrinsics.load(C.intrinsics_path(cam))
        und = cv2.undistort(img, intr.K_np, intr.dist_np)
        r = mujoco.Renderer(m, 1080, 1920)
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0
        r.update_scene(d, camera=cam, scene_option=opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        sim_img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        # outline of the sim cube only: render a mask by colour-keying the cube's geoms
        seg_opt = mujoco.MjvOption()
        seg_opt.geomgroup[3] = 0
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=cam, scene_option=seg_opt)
        seg = r.render()[..., 0]
        r.disable_segmentation_rendering()
        cube_ids = {m.geom("task_cube_body").id, m.geom("task_cube_red_face").id}
        mask = np.isin(seg, list(cube_ids)).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = cv2.addWeighted(und, 0.7, sim_img, 0.3, 0)
        cv2.drawContours(vis, cnts, -1, (255, 0, 255), 3)
        ys, xs = np.nonzero(mask)
        if len(xs):
            x0, x1 = max(0, xs.min() - 250), min(1920, xs.max() + 250)
            y0, y1 = max(0, ys.min() - 180), min(1080, ys.max() + 180)
            vis = vis[y0:y1, x0:x1]
        vis = cv2.resize(vis, (800, int(800 * vis.shape[0] / vis.shape[1])))
        cv2.putText(vis, cam, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        tiles.append(vis)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    cv2.imwrite(str(folder / "pose_check.jpg"), np.hstack(tiles))
    print("wrote", folder / "pose_check.jpg")


if __name__ == "__main__":
    main()
