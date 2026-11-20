"""Extrinsic calibration: where each camera is in the MuJoCo world, from images of the
board at a measured placement (rig.yaml). Writes the calibrated MuJoCo scene.

    python real2sim/cameras/calibrate_extrinsics.py extr                       # one session, all cameras
    python real2sim/cameras/calibrate_extrinsics.py --cam-session top=extr_top --cam-session side=extr_side
    python real2sim/cameras/calibrate_extrinsics.py --cam-session top=extr_top --via side=top:extr_pair

The --via form needs no tape measure for `side`: the board is put (e.g. leaning at ~45 deg)
where both cameras see it, and its world pose comes from the already calibrated `top`.

Each camera uses its own board placement if rig.yaml gives cameras.<name>.board_in_world
(e.g. flat on the bench for the top camera, standing up facing the side camera), else
the top-level board_in_world. The cameras therefore do not need to see the board at the
same time. Reads real2sim/data/<session>/<cam>/*.png (a few frames of the same static
setup; they are averaged), plus calib/<cam>_intrinsics.json.
Writes:
  real2sim/cameras/calib/cameras.json                     poses + intrinsics, for analysis code
  src/orca_sim/scenes/v2/<scene>_real2sim.xml             base scene + board + cameras
  src/orca_sim/scenes/v2/real2sim_board.png               board texture used by that scene
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import camlib as C


def camera_pose(image_dir: Path, intr: C.Intrinsics, det: C.BoardDetector, T_world_board: np.ndarray,
                *, min_corners: int = 10, verbose: bool = True) -> tuple[np.ndarray, dict]:
    Ts, errs, n_corners = [], [], []
    for p in sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")):
        im = cv2.imread(str(p))
        corners, ids = det.detect(im)
        if len(ids) < min_corners:
            if verbose:
                print(f"  {p.name}: only {len(ids)} corners, skipped")
            continue
        obj = det.object_points(ids)
        ok, rv, tv = cv2.solvePnP(obj, corners, intr.K_np, intr.dist_np, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        rv, tv = cv2.solvePnPRefineLM(obj, corners, intr.K_np, intr.dist_np, rv, tv)
        proj, _ = cv2.projectPoints(obj, rv, tv, intr.K_np, intr.dist_np)
        errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1)))))
        n_corners.append(len(ids))
        T_cam_board = C.make_T(cv2.Rodrigues(rv)[0], tv)
        Ts.append(T_world_board @ C.inv_T(T_cam_board))
    if not Ts:
        raise SystemExit(f"no usable board view in {image_dir}")
    R = C.average_rotations([T[:3, :3] for T in Ts])
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    T = C.make_T(R, t)
    spread_mm = max(np.linalg.norm(Ti[:3, 3] - t) for Ti in Ts) * 1000
    spread_deg = max(C.rot_angle_deg(Ti[:3, :3], R) for Ti in Ts)
    stats = {"n_frames": len(Ts), "reproj_rms_px": float(np.mean(errs)), "corners": int(np.mean(n_corners)),
             "spread_mm": float(spread_mm), "spread_deg": float(spread_deg)}
    return T, stats


def _board_in_cam(im: np.ndarray, intr: C.Intrinsics, det: C.BoardDetector, min_corners: int):
    corners, ids = det.detect(im)
    if len(ids) < min_corners:
        return None, len(ids), None
    obj = det.object_points(ids)
    ok, rv, tv = cv2.solvePnP(obj, corners, intr.K_np, intr.dist_np, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, len(ids), None
    rv, tv = cv2.solvePnPRefineLM(obj, corners, intr.K_np, intr.dist_np, rv, tv)
    proj, _ = cv2.projectPoints(obj, rv, tv, intr.K_np, intr.dist_np)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
    return C.make_T(cv2.Rodrigues(rv)[0], tv), len(ids), err


def camera_pose_via(session: Path, cam: str, intr: C.Intrinsics, ref: str, ref_intr: C.Intrinsics,
                    T_world_ref: np.ndarray, det: C.BoardDetector, *, min_corners: int = 10,
                    verbose: bool = True) -> tuple[np.ndarray, dict]:
    """Pose of `cam` from frames where an already calibrated camera `ref` sees the same board.

    No tape measure: the board's world pose comes from `ref` (T_world_ref @ T_ref_board), so
    the board can be anywhere both cameras see it (e.g. leaning at ~45 deg). capture.py saves
    both cameras' frames of one SPACE press under the same file name, which pairs them."""
    Ts, errs, n_corners = [], [], []
    for p in sorted((session / cam).glob("*.png")):
        q = session / ref / p.name
        if not q.exists():
            continue
        T_cb, n_c, e_c = _board_in_cam(cv2.imread(str(p)), intr, det, min_corners)
        T_rb, n_r, e_r = _board_in_cam(cv2.imread(str(q)), ref_intr, det, min_corners)
        if T_cb is None or T_rb is None:
            if verbose:
                print(f"  {p.name}: {cam} {n_c} / {ref} {n_r} corners, skipped")
            continue
        errs.append(max(e_c, e_r))
        n_corners.append(min(n_c, n_r))
        Ts.append(T_world_ref @ T_rb @ C.inv_T(T_cb))
    if not Ts:
        raise SystemExit(f"no frame in {session} where both {cam} and {ref} see the board")
    R = C.average_rotations([T[:3, :3] for T in Ts])
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    T = C.make_T(R, t)
    stats = {"n_frames": len(Ts), "reproj_rms_px": float(np.mean(errs)), "corners": int(np.mean(n_corners)),
             "spread_mm": float(max(np.linalg.norm(Ti[:3, 3] - t) for Ti in Ts) * 1000),
             "spread_deg": float(max(C.rot_angle_deg(Ti[:3, :3], R) for Ti in Ts)), "via": ref}
    return T, stats


def run(session: Path | None, rig: dict, *, cam_sessions: dict[str, Path] | None = None,
        via: dict[str, tuple[str, Path]] | None = None,
        write_scene: bool = True, scene_name: str | None = None,
        texture_name: str = C.BOARD_TEXTURE_NAME, verbose: bool = True) -> dict:
    """via: {cam: (ref_cam, session)} calibrates `cam` from shared views with `ref_cam`
    (see camera_pose_via) instead of from a measured board placement."""
    board_cfg = rig["board"]
    det = C.BoardDetector(board_cfg)
    cam_sessions = cam_sessions or {}
    via = via or {}
    cams, report, poses = [], {}, {}
    order = [n for n in rig["cameras"] if n not in via] + [n for n in rig["cameras"] if n in via]
    for name in order:
        intr = C.Intrinsics.load(C.intrinsics_path(name))
        if name in via:
            ref, sess = via[name]
            if ref not in poses:
                raise SystemExit(f"{name} is calibrated via {ref}, but {ref} has no pose")
            biw = None
            T, stats = camera_pose_via(sess, name, intr, ref, C.Intrinsics.load(C.intrinsics_path(ref)),
                                       poses[ref], det, verbose=verbose)
        else:
            sess = cam_sessions.get(name, session)
            if sess is None:
                raise SystemExit(f"no session given for camera {name}")
            biw = C.board_pose_for(rig, name)
            T, stats = camera_pose(sess / name, intr, det, C.T_world_boardcv(board_cfg, biw), verbose=verbose)
        poses[name] = T
        cams.append(C.MjCamera.from_intrinsics(name, T, intr))
        R_mj = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        report[name] = {
            "T_world_cam_opencv": T.tolist(),
            "pos_m": T[:3, 3].tolist(),
            "look_dir_world": T[:3, 2].tolist(),
            "mujoco_xyaxes": np.concatenate([R_mj[:, 0], R_mj[:, 1]]).tolist(),
            "intrinsics": {"width": intr.width, "height": intr.height, "K": intr.K, "dist": intr.dist},
            "board_in_world": biw,
            "session": str(sess),
            **stats,
        }
        if verbose:
            p = T[:3, 3]
            print(f"{name}: pos ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m   looking {np.round(T[:3, 2], 3)}   "
                  f"{stats['n_frames']} frames, reproj {stats['reproj_rms_px']:.2f} px, "
                  f"spread {stats['spread_mm']:.2f} mm / {stats['spread_deg']:.3f} deg")
    out = {"scene": rig["scene"], "board": board_cfg, "board_in_world": rig["board_in_world"], "cameras": report}
    if write_scene:
        C.CALIB_DIR.mkdir(parents=True, exist_ok=True)
        (C.CALIB_DIR / "cameras.json").write_text(json.dumps(out, indent=2))
        C.write_board_texture(board_cfg, texture_name)
        scene_name = scene_name or rig["scene"].replace(".xml", "_real2sim.xml")
        T_wb = C.T_world_boardcv(board_cfg, rig["board_in_world"])
        xml = C.build_scene_xml(rig["scene"], board_cfg, T_wb, cams, texture_file=texture_name,
                                model_name=scene_name.removesuffix(".xml"), bench=rig.get("bench"),
                                ultrasonic=rig.get("ultrasonic"))
        (C.SCENES_DIR / scene_name).write_text(xml)
        out["scene_file"] = str(C.SCENES_DIR / scene_name)
        if verbose:
            print(f"wrote {C.CALIB_DIR / 'cameras.json'} and {C.SCENES_DIR / scene_name}")
    return out


def write_scene(rig: dict, cal: dict, *, scene_name: str | None = None,
                texture_name: str = C.BOARD_TEXTURE_NAME) -> Path:
    """(Re)write the _real2sim scene from a cameras.json-style dict (e.g. after correct_bump.py)."""
    cams = [C.MjCamera.from_intrinsics(n, np.asarray(c["T_world_cam_opencv"]), C.Intrinsics.load(C.intrinsics_path(n)))
            for n, c in cal["cameras"].items()]
    C.write_board_texture(rig["board"], texture_name)
    scene_name = scene_name or rig["scene"].replace(".xml", "_real2sim.xml")
    xml = C.build_scene_xml(rig["scene"], rig["board"], C.T_world_boardcv(rig["board"], rig["board_in_world"]), cams,
                            texture_file=texture_name, model_name=scene_name.removesuffix(".xml"),
                            bench=rig.get("bench"), ultrasonic=rig.get("ultrasonic"))
    path = C.SCENES_DIR / scene_name
    path.write_text(xml)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="folder under real2sim/data/ (or a path) used for all cameras")
    ap.add_argument("--cam-session", action="append", default=[], metavar="CAM=SESSION",
                    help="per-camera session, e.g. side=extr_side (overrides `session` for that camera)")
    ap.add_argument("--via", action="append", default=[], metavar="CAM=REF:SESSION",
                    help="calibrate CAM from frames shared with the calibrated camera REF, e.g. side=top:extr_pair "
                         "(no board measurement needed for CAM)")
    ap.add_argument("--rig", default=C.HERE / "rig.yaml")
    args = ap.parse_args()

    def resolve(s: str) -> Path:
        p = Path(s)
        return p if p.exists() else C.DATA_DIR / s

    per_cam = {}
    for item in args.cam_session:
        cam, _, s = item.partition("=")
        per_cam[cam] = resolve(s)
    via = {}
    for item in args.via:
        cam, _, rest = item.partition("=")
        ref, _, s = rest.partition(":")
        via[cam] = (ref, resolve(s))
    run(resolve(args.session) if args.session else None, C.load_rig(args.rig), cam_sessions=per_cam, via=via)


if __name__ == "__main__":
    main()
