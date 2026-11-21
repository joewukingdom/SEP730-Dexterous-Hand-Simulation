"""Annotated key moments of a logged policy-replay recording, in plain words (Chinese labels).

    python real2sim/recording/keyframes.py run07 policy_run07.yaml run07

Picks four moments automatically: the start, when the cube starts to move (camera pose more
than 3 mm / 5 deg from the start), the last frame with the cube on the hand, and 2 s after.
Each row: top and side frames with the sim hand (replayed angles, magenta) and the sim cube
(cyan) drawn over them, and a caption: real and sim ultrasonic, what the sim beam hits, how
far the cube has moved / turned. Needs dataset_twin.csv (twin.py).
Writes docs/figures/real2sim/<name>_keyframes.jpg.
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
import record as REC  # noqa: E402  (Chinese text drawing)
import twin as W  # noqa: E402

C = W.C
HIT = {"cube": "方块", "hand": "手指/手掌", "bench": "桌面", "none": "没有回波"}


def quat_angle(q0, q1) -> float:
    d = abs(float(np.dot(q0, q1)))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def main() -> None:
    name, replay_name, log_name = sys.argv[1:4]
    rec = C.DATA_DIR / "recordings" / name
    rows = list(csv.DictReader(open(rec / "dataset_twin.csv", newline="")))
    rep = yaml.safe_load((W.ORCA / "replay_sequences" / replay_name).read_text(encoding="utf-8"))
    angles = np.asarray(rep["angles"], float)
    ids = rep["metadata"]["joint_ids"]
    meta = json.loads((rec / "meta.json").read_text())
    log = np.array([float(r["perf_counter"]) - meta["perf_counter_origin"]
                    for r in csv.DictReader(open(C.DATA_DIR / "replay_logs" / f"{log_name}.csv", newline=""))])
    found = [r for r in rows if r["cube_found"] == "1"]
    p0 = np.array([float(found[0][k]) for k in ("cube_x", "cube_y", "cube_z")])
    q0 = np.array([float(found[0][k]) for k in ("cube_qw", "cube_qx", "cube_qy", "cube_qz")])

    def moved(r):
        p = np.array([float(r[k]) for k in ("cube_x", "cube_y", "cube_z")])
        q = np.array([float(r[k]) for k in ("cube_qw", "cube_qx", "cube_qy", "cube_qz")])
        return np.linalg.norm(p - p0) * 1000, quat_angle(q0, q)

    on_hand = [r for r in found if moved(r)[0] < 60]  # far jumps = the cube has left the hand
    start = rows[min(5, len(rows) - 1)]
    first_move = next((r for r in on_hand if moved(r)[0] > 3 or moved(r)[1] > 5), on_hand[-1])
    last_on = on_hand[-1]
    t_after = float(last_on["t"]) + 2.0
    after = min(rows, key=lambda r: abs(float(r["t"]) - t_after))
    picks = [("开始", start), ("方块开始被推动", first_move), ("掉落前最后一帧", last_on), ("掉落 2 秒后", after)]

    rig = C.load_rig(C.HERE / "rig.yaml")
    m = mujoco.MjModel.from_xml_path(str((C.SCENES_DIR / rig["scene"].replace(".xml", "_real2sim.xml")).resolve()))
    d = mujoco.MjData(m)
    bridge = W.JointBridge(m, yaml.safe_load(W.CONFIG.read_text(encoding="utf-8"))["joint_ids"])
    cj = m.joint("cube_freejoint").qposadr[0]
    cube_body = m.body("task_cube").id
    cube_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == cube_body]
    hand_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] not in (0, cube_body)]
    r_ = mujoco.Renderer(m, 1080, 1920)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    intr = {c: C.Intrinsics.load(C.intrinsics_path(c)) for c in ("top", "side")}
    t0 = float(rows[0]["t"])
    out_rows = []
    for title, r in picks:
        t = float(r["t"])
        k = int(np.clip(np.searchsorted(log, t, side="right") - 1, 0, len(angles) - 1))
        mujoco.mj_resetData(m, d)
        W.set_sim_pose(bridge, d.qpos, dict(zip(ids, angles[k])))
        if r["cube_found"] == "1":
            d.qpos[cj:cj + 3] = [float(r[c]) for c in ("cube_x", "cube_y", "cube_z")]
            d.qpos[cj + 3:cj + 7] = [float(r[c]) for c in ("cube_qw", "cube_qx", "cube_qy", "cube_qz")]
        else:
            d.qpos[cj:cj + 3] = [0.0, 0.0, -1.0]
        mujoco.mj_forward(m, d)
        tiles = []
        for cam in ("top", "side"):
            und = cv2.undistort(cv2.imread(str(rec / r[f"{cam}_file"])), intr[cam].K_np, intr[cam].dist_np)
            r_.enable_segmentation_rendering()
            r_.update_scene(d, camera=cam, scene_option=opt)
            seg = r_.render()[..., 0]
            r_.disable_segmentation_rendering()
            for geoms, col in ((hand_geoms, (255, 0, 255)), (cube_geoms, (255, 255, 0))):
                cnts, _ = cv2.findContours(np.isin(seg, geoms).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(und, cnts, -1, col, 3)
            crop = und[100:1000, 650:1450] if cam == "top" else und[250:850, 750:1900]
            tiles.append(cv2.resize(crop, (int(380 * crop.shape[1] / crop.shape[0]), 380)))
        row = np.hstack(tiles)
        real = f"{float(r['us_mm']):.0f} mm" if r["us_mm"] else "无回波"
        sim = f"{float(r['sim_twin_mm']):.0f} mm" if r["sim_twin_mm"] else "无回波"
        if r["cube_found"] == "1" and r in on_hand:
            dm, da = moved(r)
            cube_txt = f"方块相对开始：移动 {dm:.0f} mm，转动 {da:.0f}°"
        elif r["cube_found"] == "1":
            cube_txt = "方块已离开手（相机在手外检测到它）"
        else:
            cube_txt = "相机没看到方块"
        band = np.full((150, row.shape[1], 3), 30, np.uint8)
        band = REC._text(band, [(f"{title}（t = {t - t0:.1f} s）", 0, (0, 220, 255)),
                                (f"超声波 真实 {real} ｜ sim {sim}（sim 打到：{HIT[r['sim_twin_hit']]}）", 0,
                                 (255, 255, 255)), (cube_txt, 0, (200, 200, 200))], 12, 6)
        out_rows.append(np.vstack([band, row]))
    wmax = max(x.shape[1] for x in out_rows)
    out_rows = [cv2.copyMakeBorder(x, 0, 0, 0, wmax - x.shape[1], cv2.BORDER_CONSTANT) for x in out_rows]
    sheet = np.vstack(out_rows)
    foot = REC._text(np.full((50, wmax, 3), 30, np.uint8),
                     [("洋红 = sim 手（按回放指令），青色 = sim 方块（按相机测得的姿态）", 0, (255, 255, 255))], 12, 8)
    sheet = np.vstack([sheet, foot])
    out = C.REPO_ROOT / "docs" / "figures" / "real2sim" / f"{name}_keyframes.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out)


if __name__ == "__main__":
    main()
