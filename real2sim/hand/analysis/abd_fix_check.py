"""Ring / pinky outlines over the real frames of a corrected-mapping recording (pose set abdfix):
old mapping (JointBridge only) in blue, corrected mapping (joint_map.py) in magenta.

    python real2sim/hand/analysis/abd_fix_check.py abdcheck03
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

name = sys.argv[1]
rec = C.DATA_DIR / "recordings" / name
sched = json.load(open(REPO / "real2sim" / "hand" / "pose_check_abdfix_schedule.json"))
off = json.load(open(rec / "pose_check_scores.json"))["replay_offset_s"]
cfg = yaml.safe_load(open(r"C:\Users\Owner\Desktop\orca\orca_core\models\v2\orcahand_right\config.yaml", encoding="utf-8"))
rig = C.load_rig(C.HERE / "rig.yaml")
m = mujoco.MjModel.from_xml_path(str((C.SCENES_DIR / rig["scene"].replace(".xml", "_real2sim.xml")).resolve()))
d = mujoco.MjData(m)
br = JointBridge(m, cfg["joint_ids"])
cj = m.joint("cube_freejoint").qposadr[0]
r = mujoco.Renderer(m, 1080, 1920)
opt = mujoco.MjvOption()
opt.geomgroup[3] = 0
intr = C.Intrinsics.load(C.intrinsics_path("top"))
frames = [(float(x["t"]), x["file"]) for x in csv.DictReader(open(rec / "top_frames.csv"))]


def finger_geoms(joint):
    qadr = br.qadr_of_core[joint]
    jid = int(np.where(m.jnt_qposadr == qadr)[0][0])
    root = int(m.jnt_bodyid[jid])
    bodies = set()
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p == root:
                bodies.add(b)
                break
            p = int(m.body_parentid[p])
    return {g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies}


geoms = finger_geoms("ring_abd") | finger_geoms("pinky_abd")
tiles = []
for p in sched["poses"]:
    if not (p["pose"].startswith(("ring", "pinky")) or p["pose"] == "zero"):
        continue
    t = off + (p["hold_start_s"] + p["hold_end_s"]) / 2
    _, f = min(frames, key=lambda x: abs(x[0] - t))
    und = cv2.undistort(cv2.imread(str(rec / f)), intr.K_np, intr.dist_np)
    for corrected, col in ((False, (255, 120, 0)), (True, (255, 0, 255))):
        mujoco.mj_resetData(m, d)
        for j, deg in p["angles_deg"].items():
            v = core_to_sim_deg(j, deg) if corrected else (-deg if j == "index_abd" else deg)
            d.qpos[br.qadr_of_core[j]] = np.radians(v)
        d.qpos[cj:cj + 3] = [0, 0, -1]
        mujoco.mj_forward(m, d)
        r.enable_segmentation_rendering()
        r.update_scene(d, camera="top", scene_option=opt)
        seg = r.render()[..., 0]
        r.disable_segmentation_rendering()
        mask = np.isin(seg, list(geoms)).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(und, cnts, -1, col, 3)
    crop = cv2.resize(und[150:850, 450:1250], (400, 350))
    cv2.putText(crop, p["pose"], (6, 22), 0, 0.6, (255, 255, 255), 2)
    tiles.append(crop)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
cv2.putText(sheet, "ring + pinky outlines: blue = old mapping, magenta = corrected", (6, sheet.shape[0] - 10), 0,
            0.6, (255, 255, 255), 2)
out = REPO / "docs" / "figures" / "real2sim" / f"{name}_abd_fix.jpg"
cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
print("wrote", out)
