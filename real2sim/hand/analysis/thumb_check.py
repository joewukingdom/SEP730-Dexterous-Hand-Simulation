"""Thumb joints, one at a time: sim thumb outlines over the real frames (top and side).

    python real2sim/hand/analysis/thumb_check.py thumbcheck01 thumbcheck01   # recording, replay log

For each pose of the "thumb" set, the sim thumb is drawn at the commanded angle (blue,
current mapping) and at the negated angle (magenta): if magenta fits the real thumb better,
that joint's direction is reversed between sim and real (as ring_abd / pinky_abd were).
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "real2sim" / "cameras"))
sys.path.insert(0, str(REPO / "real2sim" / "hand"))
sys.path.insert(0, r"C:\Users\Owner\Desktop\orca")
import camlib as C  # noqa: E402
from joint_map import core_to_sim_deg  # noqa: E402
from sim_to_real import JointBridge  # noqa: E402

name, log_name = sys.argv[1], sys.argv[2]
rec = C.DATA_DIR / "recordings" / name
sched = json.load(open(REPO / "real2sim" / "hand" / "pose_check_thumb_schedule.json"))
meta = json.loads((rec / "meta.json").read_text())
first = next(csv.DictReader(open(C.DATA_DIR / "replay_logs" / f"{log_name}.csv", newline="")))
off = float(first["perf_counter"]) - meta["perf_counter_origin"]
cfg = yaml.safe_load(open(r"C:\Users\Owner\Desktop\orca\orca_core\models\v2\orcahand_right\config.yaml", encoding="utf-8"))
rig = C.load_rig(C.HERE / "rig.yaml")
m = mujoco.MjModel.from_xml_path(str((C.SCENES_DIR / rig["scene"].replace(".xml", "_real2sim.xml")).resolve()))
d = mujoco.MjData(m)
br = JointBridge(m, cfg["joint_ids"])
cj = m.joint("cube_freejoint").qposadr[0]
r = mujoco.Renderer(m, 1080, 1920)
opt = mujoco.MjvOption()
opt.geomgroup[3] = 0
intr = {c: C.Intrinsics.load(C.intrinsics_path(c)) for c in ("top", "side")}
frames = {c: [(float(x["t"]), x["file"]) for x in csv.DictReader(open(rec / f"{c}_frames.csv"))] for c in ("top", "side")}

# the whole thumb: every body below the thumb_cmc joint's body
qadr = br.qadr_of_core["thumb_cmc"]
root = int(m.jnt_bodyid[int(np.where(m.jnt_qposadr == qadr)[0][0])])
bodies = set()
for b in range(m.nbody):
    p = b
    while p > 0:
        if p == root:
            bodies.add(b)
            break
        p = int(m.body_parentid[p])
thumb = {g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies}
crops = {"top": (150, 900, 700, 1450), "side": (300, 800, 900, 1600)}  # y0, y1, x0, x1

rows = []
for p in sched["poses"]:
    if not p["pose"].startswith("thumb_") and p["pose"] != "zero":
        continue
    joint = next((j for j in ("thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip") if p["pose"].startswith(j)), None)
    t = off + (p["hold_start_s"] + p["hold_end_s"]) / 2
    tiles = []
    for cam in ("top", "side"):
        _, f = min(frames[cam], key=lambda x: abs(x[0] - t))
        und = cv2.undistort(cv2.imread(str(rec / f)), intr[cam].K_np, intr[cam].dist_np)
        for flip, col in ((False, (255, 120, 0)), (True, (255, 0, 255))):
            if flip and joint is None:
                continue
            mujoco.mj_resetData(m, d)
            for j, deg in p["angles_deg"].items():
                v = core_to_sim_deg(j, deg)
                if flip and j == joint:
                    v = -v
                d.qpos[br.qadr_of_core[j]] = np.radians(v)
            d.qpos[cj:cj + 3] = [0, 0, -1]
            mujoco.mj_forward(m, d)
            r.enable_segmentation_rendering()
            r.update_scene(d, camera=cam, scene_option=opt)
            seg = r.render()[..., 0]
            r.disable_segmentation_rendering()
            cnts, _ = cv2.findContours(np.isin(seg, list(thumb)).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(und, cnts, -1, col, 3)
        y0, y1, x0, x1 = crops[cam]
        tile = cv2.resize(und[y0:y1, x0:x1], (420, 300))
        cv2.putText(tile, f"{p['pose']} {cam}", (6, 22), 0, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
    rows.append(np.hstack(tiles))
sheet = np.vstack(rows)
cv2.putText(sheet, "sim thumb: blue = commanded (current mapping), magenta = that joint negated",
            (6, sheet.shape[0] - 10), 0, 0.6, (255, 255, 255), 2)
out = REPO / "docs" / "figures" / "real2sim" / f"{name}_thumb.jpg"
cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
print("wrote", out)
