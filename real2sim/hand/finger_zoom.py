"""Close-ups of a pose check: for every pose that moves one joint, crop both camera views
around the finger that moves and draw only that finger's sim outline (magenta) over the
real frame, so a wrong direction / offset is visible by eye.

    python real2sim/hand/finger_zoom.py fullcheck01 full fullcheck01      # recording, set, replay log

Writes real2sim/data/recordings/<rec>/zoom_<finger>.jpg (one sheet per finger: rows =
zero pose + the two angles of each of its joints).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cameras"))
import pose_check as P  # noqa: E402

sys.path.insert(0, str(P.ORCA))
import camlib as C  # noqa: E402
from joint_map import set_sim_pose  # noqa: E402
from sim_to_real import JointBridge  # noqa: E402

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def main() -> None:
    name, pose_set, log_name = sys.argv[1:4]
    rec = C.DATA_DIR / "recordings" / name
    sched = json.loads(P.schedule_path(pose_set).read_text())
    cfg = yaml.safe_load(P.CONFIG.read_text(encoding="utf-8"))
    meta = json.loads((rec / "meta.json").read_text())
    first = next(csv.DictReader(open(C.DATA_DIR / "replay_logs" / f"{log_name}.csv", newline="")))
    off = float(first["perf_counter"]) - meta["perf_counter_origin"]
    frames = {c: [(float(r["t"]), r["file"]) for r in csv.DictReader(open(rec / f"{c}_frames.csv", newline=""))]
              for c in ("top", "side")}
    rig = C.load_rig(C.HERE / "rig.yaml")
    m = mujoco.MjModel.from_xml_path(str((C.SCENES_DIR / rig["scene"].replace(".xml", "_real2sim.xml")).resolve()))
    d = mujoco.MjData(m)
    bridge = JointBridge(m, cfg["joint_ids"])
    cj = m.joint("cube_freejoint").qposadr[0]
    intr = {c: C.Intrinsics.load(C.intrinsics_path(c)) for c in ("top", "side")}
    r = mujoco.Renderer(m, 1080, 1920)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0

    def subtree_geoms(joint: str) -> list[int]:
        """Geoms of the finger the joint belongs to (subtree of the finger's first joint)."""
        finger = joint.split("_")[0]
        roots = [bridge.qadr_of_core[j] for j in cfg["joint_ids"] if j.startswith(finger + "_")]
        jids = [int(np.where(m.jnt_qposadr == q)[0][0]) for q in roots]
        root = min(int(m.jnt_bodyid[j]) for j in jids)
        bodies = set()
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p == root:
                    bodies.add(b)
                    break
                p = int(m.body_parentid[p])
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]

    def render(p, cam, geoms):
        mujoco.mj_resetData(m, d)
        set_sim_pose(bridge, d.qpos, p["angles_deg"])
        d.qpos[cj:cj + 3] = [0.0, 0.0, -1.0]
        mujoco.mj_forward(m, d)
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=cam, scene_option=opt)
        seg = r.render()[..., 0]
        r.disable_segmentation_rendering()
        return np.isin(seg, geoms).astype(np.uint8)

    zero = next(p for p in sched["poses"] if p["pose"] == "zero")
    for finger in FINGERS:
        poses = [zero] + [p for p in sched["poses"] if p["pose"].startswith(finger + "_")]
        geoms = subtree_geoms(poses[-1]["pose"].rsplit("_", 1)[0])
        # one crop box per camera covering the finger in every pose
        boxes = {}
        masks = {}
        for cam in ("top", "side"):
            ms = [render(p, cam, geoms) for p in poses]
            masks[cam] = ms
            ys, xs = np.nonzero(np.max(ms, axis=0))
            pad = 70
            boxes[cam] = (max(0, xs.min() - pad), min(1920, xs.max() + pad), max(0, ys.min() - pad),
                          min(1080, ys.max() + pad))
        rows = []
        for i, p in enumerate(poses):
            t_mid = off + (p["hold_start_s"] + p["hold_end_s"]) / 2
            tiles = []
            for cam in ("top", "side"):
                _, f = min(frames[cam], key=lambda x: abs(x[0] - t_mid))
                und = cv2.undistort(cv2.imread(str(rec / f)), intr[cam].K_np, intr[cam].dist_np)
                cnts, _ = cv2.findContours(masks[cam][i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(und, cnts, -1, (255, 0, 255), 2)
                x0, x1, y0, y1 = boxes[cam]
                crop = und[y0:y1, x0:x1]
                tiles.append(cv2.resize(crop, (int(360 * crop.shape[1] / crop.shape[0]), 360)))
            row = np.hstack(tiles)
            cv2.putText(row, p["pose"], (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
            cv2.putText(row, p["pose"], (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            rows.append(row)
        sheet = np.vstack(rows)
        out = rec / f"zoom_{finger}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", out, sheet.shape)


if __name__ == "__main__":
    main()
