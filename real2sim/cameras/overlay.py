"""Real-vs-sim check for the calibrated cameras.

    python real2sim/cameras/overlay.py --calib                 # the frames used for calibration (sanity)
    python real2sim/cameras/overlay.py val_side --cams side --board-pos 0.12 0.15 0.05 --board-tilt -90

For every camera: undistort the real frame (so it becomes a pinhole image with the same
K as the sim camera), render the calibrated MuJoCo camera with the board at the given
pose, detect the board corners in BOTH images and compare them id by id.

Reports mean / max corner error in pixels and in mm on the board, and saves
calib/overlay_<cam>.png: 50/50 blend, real corners green, sim corners red.

Use a held-out placement for the real number: move the board to a second measured
position (pass it with --board-pos / --board-yaw) that was NOT used by
calibrate_extrinsics.py. The hand is rendered at the scene's default pose, so put the
real hand in its zero pose when you capture if you also want the hand outline to line up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import camlib as C


def load_scene(path: Path):
    m = mujoco.MjModel.from_xml_path(str(path.resolve()))
    d = mujoco.MjData(m)
    return m, d


def set_board_pose(m, d, T_world_board: np.ndarray) -> None:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "real2sim_board")
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(T_world_board[:3, :3]).ravel())
    if m.body_mocapid[bid] >= 0:
        d.mocap_pos[m.body_mocapid[bid]] = T_world_board[:3, 3]
        d.mocap_quat[m.body_mocapid[bid]] = q
    else:
        m.body_pos[bid] = T_world_board[:3, 3]
        m.body_quat[bid] = q
    mujoco.mj_forward(m, d)


def compare(real_bgr: np.ndarray, sim_bgr: np.ndarray, det: C.BoardDetector, K: np.ndarray,
            T_world_cam: np.ndarray, T_world_board: np.ndarray) -> dict:
    rc, rid = det.detect(real_bgr)
    sc, sid = det.detect(sim_bgr)
    common = np.intersect1d(rid, sid)
    if len(common) == 0:
        return {"n": 0, "real_corners": len(rid), "sim_corners": len(sid)}
    real_by_id = dict(zip(rid.tolist(), rc))
    sim_by_id = dict(zip(sid.tolist(), sc))
    r = np.array([real_by_id[i] for i in common.tolist()])
    s = np.array([sim_by_id[i] for i in common.tolist()])
    err = np.linalg.norm(r - s, axis=1)
    # convert to mm on the board plane using each corner's depth from the camera
    obj = det.object_points(common)
    pw = (T_world_board[:3, :3] @ obj.T).T + T_world_board[:3, 3]
    Tcw = C.inv_T(T_world_cam)
    depth = ((Tcw[:3, :3] @ pw.T).T + Tcw[:3, 3])[:, 2]
    err_mm = err * depth / K[0, 0] * 1000
    return {"n": int(len(common)), "real_corners": int(len(rid)), "sim_corners": int(len(sid)),
            "mean_px": float(err.mean()), "max_px": float(err.max()),
            "mean_mm": float(err_mm.mean()), "max_mm": float(err_mm.max()),
            "real_pts": r, "sim_pts": s}


def draw(real_bgr, sim_bgr, res) -> np.ndarray:
    vis = cv2.addWeighted(real_bgr, 0.5, sim_bgr, 0.5, 0)
    if res.get("n"):
        for a, b in zip(res["real_pts"], res["sim_pts"]):
            cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 255), 2)
            cv2.circle(vis, tuple(map(int, a)), 6, (0, 255, 0), -1)
            cv2.circle(vis, tuple(map(int, b)), 4, (0, 0, 255), -1)
        txt = f"{res['n']} corners  mean {res['mean_px']:.2f}px ({res['mean_mm']:.2f}mm)  max {res['max_px']:.2f}px"
    else:
        txt = f"no common corners (real {res['real_corners']}, sim {res['sim_corners']})"
    cv2.putText(vis, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    return vis


def run(session: Path | None, *, scene_file: Path | None = None, board_in_world: dict | None = None,
        cams: list[str] | None = None, index: int = 0, show: bool = True, save: bool = True,
        verbose: bool = True) -> dict:
    """session=None: use each camera's own calibration session and board placement."""
    cal = json.loads((C.CALIB_DIR / "cameras.json").read_text())
    board_cfg = cal["board"]
    det = C.BoardDetector(board_cfg)
    scene_file = scene_file or C.SCENES_DIR / cal["scene"].replace(".xml", "_real2sim.xml")
    m, d = load_scene(scene_file)
    results = {}
    for name, cam in cal["cameras"].items():
        if cams and name not in cams:
            continue
        sess = session if session is not None else Path(cam["session"])
        T_wb = C.T_world_boardcv(board_cfg, board_in_world or cam["board_in_world"])
        set_board_pose(m, d, T_wb)
        files = sorted((sess / name).glob("*.png")) + sorted((sess / name).glob("*.jpg"))
        if not files:
            print(f"{name}: no images in {sess / name}")
            continue
        real = cv2.imread(str(files[min(index, len(files) - 1)]))
        K = np.asarray(cam["intrinsics"]["K"])
        real_u = cv2.undistort(real, K, np.asarray(cam["intrinsics"]["dist"]))
        sim = C.render_camera(m, d, name)
        res = compare(real_u, sim, det, K, np.asarray(cam["T_world_cam_opencv"]), T_wb)
        vis = draw(real_u, sim, res)
        if save:
            cv2.imwrite(str(C.CALIB_DIR / f"overlay_{name}.png"), vis)
        if verbose:
            if res.get("n"):
                print(f"{name}: {res['n']} corners  mean {res['mean_px']:.2f} px = {res['mean_mm']:.2f} mm   "
                      f"max {res['max_px']:.2f} px = {res['max_mm']:.2f} mm")
            else:
                print(f"{name}: no common corners (real {res['real_corners']}, sim {res['sim_corners']})")
        if show:
            cv2.imshow(f"overlay {name} (any key)", cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2)))
        results[name] = {k: v for k, v in res.items() if not k.endswith("_pts")}
    if show and results:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="folder under real2sim/data/ (or a path)")
    ap.add_argument("--calib", action="store_true", help="re-check the frames used by calibrate_extrinsics.py")
    ap.add_argument("--cams", nargs="*", help="only these cameras")
    ap.add_argument("--board-pos", type=float, nargs=3, help="board centre in world (m); default: calibration placement")
    ap.add_argument("--board-yaw", type=float, default=0.0, help="deg, with --board-pos")
    ap.add_argument("--board-tilt", type=float, default=0.0, help="deg about the board's long edge, with --board-pos")
    ap.add_argument("--index", type=int, default=0, help="which frame of the session")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if args.calib == bool(args.session):
        ap.error("give either a session or --calib")
    session = None
    if args.session:
        session = Path(args.session)
        if not session.exists():
            session = C.DATA_DIR / args.session
    biw = {"pos_m": args.board_pos, "yaw_deg": args.board_yaw, "tilt_deg": args.board_tilt} if args.board_pos else None
    res = run(session, board_in_world=biw, cams=args.cams, index=args.index, show=not args.no_show)
    tag = session.name if session else "calib"
    (C.CALIB_DIR / f"overlay_{tag}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
