"""Sim finger outlines at -20 / 0 / +20 deg abduction drawn over the real frame of each pose."""
import csv
import json
import sys
import cv2
import mujoco
import numpy as np
import yaml

REPO = r"c:/Users/Owner/Desktop/ORCA project/SEP730-Dexterous-Hand-Simulation"
sys.path.insert(0, REPO + "/real2sim/cameras")
sys.path.insert(0, r"C:\Users\Owner\Desktop\orca")
import camlib as C
from sim_to_real import JointBridge

rec = C.DATA_DIR / "recordings" / "abdcheck01"
sched = json.load(open(REPO + "/real2sim/hand/pose_check_abd_schedule.json"))
off = 81.23
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


cols = {-20: (255, 80, 0), 0: (0, 255, 255), 20: (255, 0, 255)}
tiles = []
for joint in ("index_abd", "ring_abd", "pinky_abd"):
    geoms = finger_geoms(joint)
    for cmd in (-20, 20):
        p = next(q for q in sched["poses"] if q["pose"] == f"{joint}_{cmd:+d}")
        t = off + (p["hold_start_s"] + p["hold_end_s"]) / 2
        _, f = min(frames, key=lambda x: abs(x[0] - t))
        und = cv2.undistort(cv2.imread(str(rec / f)), intr.K_np, intr.dist_np)
        for sim_deg, col in cols.items():
            mujoco.mj_resetData(m, d)
            for j, deg in p["angles_deg"].items():
                v = sim_deg if j == joint else deg
                d.qpos[br.qadr_of_core[j]] = np.radians(-v if j == "index_abd" else v)
            d.qpos[cj:cj + 3] = [0, 0, -1]
            mujoco.mj_forward(m, d)
            r.enable_segmentation_rendering()
            r.update_scene(d, camera="top", scene_option=opt)
            seg = r.render()[..., 0]
            r.disable_segmentation_rendering()
            mask = np.isin(seg, list(geoms)).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(und, cnts, -1, col, 3)
        crop = cv2.resize(und[100:800, 600:1350], (375, 350))
        cv2.putText(crop, f"real {joint} {cmd:+d}", (6, 22), 0, 0.6, (255, 255, 255), 2)
        tiles.append(crop)
sheet = np.vstack([np.hstack(tiles[k:k + 2]) for k in (0, 2, 4)])
cv2.putText(sheet, "sim: blue -20  yellow 0  magenta +20", (6, sheet.shape[0] - 10), 0, 0.6, (255, 255, 255), 2)
cv2.imwrite(r"C:\Users\Owner\AppData\Local\Temp\claude\805d431afcdcfcda\6a49691b-518d-4c27-b646-1301d45b5d24\scratchpad\abd_dir.jpg", sheet)
