"""AprilTag cube: find where each tag actually sits on the cube, then estimate cube poses.

    # 1. capture: hold the cube under the top camera and turn it slowly so that two or three
    #    tags are visible at once; frames with >= 2 tags are saved automatically. Q to stop.
    python real2sim/cube/cube_tags.py capture

    # 2. solve the layout from those frames -> real2sim/cameras/calib/cube_tags_layout.json
    python real2sim/cube/cube_tags.py solve

Only tag 0 has to be on the face that plays the sim's red face (the green face). The other
tags may be on any face in any rotation, a bit off-centre: their real placement is measured
here from frames that see two tags at once (relative pose of the two tags), and the cube's
axes are then snapped to the measured face normals. Tag size and cube edge come from
targets/cube_tags.json (27 mm tags, 35 mm cube).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cameras"))
import camlib as C  # noqa: E402

TAGS = json.loads((C.HERE / "targets" / "cube_tags.json").read_text())
# the size the detector effectively sees (see effective_tag_note in cube_tags.json)
TAG_M = TAGS.get("effective_tag_mm", TAGS["tag_mm"]) / 1000.0
HALF = TAGS["face_mm"] / 2000.0
LAYOUT = C.CALIB_DIR / "cube_tags_layout.json"
CAP_DIR = C.DATA_DIR / "cube_tags"


def detector() -> cv2.aruco.ArucoDetector:
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)


def tag_poses(img: np.ndarray, intr: C.Intrinsics, det=None) -> dict[int, tuple[np.ndarray, float]]:
    """{id: (T_cam_tag, reprojection px)}; tag frame: x right, y up, z out of the tag."""
    det = det or detector()
    corners, ids, _ = det.detectMarkers(img)
    out = {}
    if ids is None:
        return out
    s = TAG_M / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    for c, i in zip(corners, ids.ravel()):
        if int(i) > 5:
            continue
        img_pts = c.reshape(4, 2).astype(np.float64)
        ok, rv, tv = cv2.solvePnP(obj, img_pts, intr.K_np, intr.dist_np, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rv, tv, intr.K_np, intr.dist_np)
        err = float(np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1).mean())
        out[int(i)] = (C.make_T(cv2.Rodrigues(rv)[0], tv.ravel()), err)
    return out


def capture(args) -> None:
    rig = C.load_rig(C.HERE / "rig.yaml")
    cam = rig["cameras"][args.cam]
    intr = C.Intrinsics.load(C.intrinsics_path(args.cam))
    out = CAP_DIR / args.cam
    out.mkdir(parents=True, exist_ok=True)
    idx = len(list(out.glob("*.png")))
    print("opening camera (can take a minute) ...", flush=True)
    cap = C.open_camera(cam["device"], cam["width"], cam["height"])
    det = detector()
    win = "cube tags  (turn the cube slowly; saved automatically when >= 2 tags; Q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last_save, seen_pairs = 0.0, {}
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        poses = tag_poses(frame, intr, det)
        vis = frame.copy()
        for i, (T, e) in poses.items():
            rv, _ = cv2.Rodrigues(T[:3, :3])
            cv2.drawFrameAxes(vis, intr.K_np, intr.dist_np, rv, T[:3, 3], TAG_M * 0.6, 3)
            uv, _ = cv2.projectPoints(np.zeros((1, 3)), rv, T[:3, 3], intr.K_np, intr.dist_np)
            cv2.putText(vis, f"id {i}", tuple(int(v) for v in uv.ravel()), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (0, 255, 255), 4)
        ids = sorted(poses)
        if len(ids) >= 2 and time.time() - last_save > args.every:
            cv2.imwrite(str(out / f"{idx:04d}.png"), frame)
            idx += 1
            last_save = time.time()
            for a in ids:
                for b in ids:
                    if a < b:
                        seen_pairs[(a, b)] = seen_pairs.get((a, b), 0) + 1
        # tags connected to id 0 directly or through other tags (id 1 is opposite id 0, so it
        # can only ever be linked through a side tag)
        linked, grow = {0}, True
        while grow:
            grow = False
            for a, b in seen_pairs:
                if (a in linked) != (b in linked):
                    linked |= {a, b}
                    grow = True
        msg = f"saved {idx}   tags now {ids}   pairs {len(seen_pairs)}   linked to id0: {sorted(linked)}"
        cv2.putText(vis, msg, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 8)
        cv2.putText(vis, msg, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.imshow(win, cv2.resize(vis, (1280, 720)))
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"{idx} frames in {out}; tag pairs seen: {dict(sorted(seen_pairs.items()))}")


def _avg_T(Ts: list[np.ndarray]) -> np.ndarray:
    return C.make_T(C.average_rotations([T[:3, :3] for T in Ts]), np.mean([T[:3, 3] for T in Ts], axis=0))


def solve(args) -> None:
    intr = C.Intrinsics.load(C.intrinsics_path(args.cam))
    det = detector()
    rel: dict[tuple[int, int], list[np.ndarray]] = {}
    files = sorted((CAP_DIR / args.cam).glob("*.png"))
    for f in files:
        poses = {i: T for i, (T, e) in tag_poses(cv2.imread(str(f)), intr, det).items() if e < args.max_err}
        for a in poses:
            for b in poses:
                if a != b:
                    rel.setdefault((a, b), []).append(C.inv_T(poses[a]) @ poses[b])  # T_a_b
    if not rel:
        raise SystemExit("no frame with two tags; run capture first")
    # T_tag0_tagi by breadth-first search over the pair graph, best-observed edges first
    T0 = {0: np.eye(4)}
    frontier = [0]
    while frontier:
        nxt = []
        for a in frontier:
            for (x, b), Ts in sorted(rel.items(), key=lambda kv: -len(kv[1])):
                if x == a and b not in T0 and len(Ts) >= 2:
                    T0[b] = T0[a] @ _avg_T(Ts)
                    nxt.append(b)
        frontier = nxt
    missing = sorted(set(range(6)) - set(T0))
    # cube frame (expressed in tag-0 frame): z = tag-0 normal; x from the side tags' normals
    n = {i: T[:3, 2] for i, T in T0.items()}
    z = n[0] / np.linalg.norm(n[0])
    sides = [i for i in n if i != 0 and abs(n[i] @ z) < 0.5]
    if not sides:
        raise SystemExit(f"no side tag linked to tag 0 (linked: {sorted(T0)}); capture frames showing "
                         "tag 0 together with a side tag")
    # snap: each side normal is +-x or +-y; average them into one x axis
    x_ref = n[sides[0]] - (n[sides[0]] @ z) * z
    x_ref /= np.linalg.norm(x_ref)
    acc = np.zeros(3)
    for i in sides:
        v = n[i] - (n[i] @ z) * z
        v /= np.linalg.norm(v)
        y_ref = np.cross(z, x_ref)
        # rotate v onto the x_ref direction by the nearest multiple of 90 deg
        k = int(np.round(np.arctan2(v @ y_ref, v @ x_ref) / (np.pi / 2)))
        ang = -k * np.pi / 2
        acc += np.cos(ang) * v + np.sin(ang) * np.cross(z, v)
    x = acc / np.linalg.norm(acc)
    y = np.cross(z, x)
    R0c = np.column_stack([x, y, z])  # cube axes in tag-0 frame
    # cube centre: each tag centre minus half an edge along its (snapped) outward normal
    centres = [T[:3, 3] - HALF * T[:3, 2] for T in T0.values()]
    c0 = np.mean(centres, axis=0)
    spread = max(np.linalg.norm(c - c0) for c in centres) * 1000
    T_tag0_cube = C.make_T(R0c, c0)
    layout, report = {}, []
    faces = {(0, 0, 1): "+z", (0, 0, -1): "-z", (1, 0, 0): "+x", (-1, 0, 0): "-x", (0, 1, 0): "+y", (0, -1, 0): "-y"}
    for i, T in sorted(T0.items()):
        T_cube_tag = C.inv_T(T_tag0_cube) @ T
        nz = T_cube_tag[:3, 2]
        axis = tuple(int(v) for v in np.round(nz))
        off = T_cube_tag[:3, 3] - HALF * np.array(axis, float)
        dev = np.degrees(np.arccos(np.clip(nz @ np.array(axis, float), -1, 1)))
        layout[str(i)] = {"face": faces.get(axis, "?"), "T_cube_tag": T_cube_tag.tolist()}
        report.append(f"  id {i}: face {faces.get(axis, '?'):>2}  normal off by {dev:4.1f} deg  "
                      f"off-centre {np.linalg.norm(off[:2] if axis[2] else off) * 1000:4.1f} mm  "
                      f"(pairs with others: {sum(len(v) for (a, b), v in rel.items() if a == i)})")
    LAYOUT.write_text(json.dumps({"tag_mm": TAGS["tag_mm"], "cube_mm": TAGS["face_mm"],
                                  "frames": len(files), "centre_spread_mm": spread, "tags": layout}, indent=2))
    print(f"{len(files)} frames; cube centre consistent to {spread:.1f} mm across tags")
    print("\n".join(report))
    if missing:
        print(f"NOT LINKED (no frame with them and a linked tag): {missing} -> capture more")
    print("wrote", LAYOUT)


def cube_pose_in_cam(poses: dict[int, tuple[np.ndarray, float]], layout: dict, max_err: float = 2.0):
    """T_cam_cube from all visible tags (rotation-averaged), plus the ids used."""
    Ts, used = [], []
    for i, (T, e) in poses.items():
        L = layout["tags"].get(str(i))
        if L is None or e > max_err:
            continue
        Ts.append(T @ C.inv_T(np.asarray(L["T_cube_tag"])))
        used.append(i)
    return (_avg_T(Ts) if Ts else None), used


def tag_corners_cube(layout: dict, i: int) -> np.ndarray:
    """The 4 corners of tag i in the cube frame (same order as the detector)."""
    s = TAG_M / 2
    obj = np.array([[-s, s, 0, 1], [s, s, 0, 1], [s, -s, 0, 1], [-s, -s, 0, 1]], np.float64)
    return (np.asarray(layout["tags"][str(i)]["T_cube_tag"]) @ obj.T).T[:, :3]


def cube_pose_joint(img: np.ndarray, intr: C.Intrinsics, layout: dict, det=None, max_err: float = 2.0):
    """T_cam_cube from ONE PnP over all corners of all visible tags (starts from the
    per-tag average). Returns (T or None, ids used, reprojection px)."""
    det = det or detector()
    corners, ids, _ = det.detectMarkers(img)
    if ids is None:
        return None, [], None
    poses = tag_poses(img, intr, det)
    T_init, used = cube_pose_in_cam(poses, layout, max_err)
    if T_init is None:
        return None, [], None
    by_id = {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel())}
    obj = np.vstack([tag_corners_cube(layout, i) for i in used])
    img_pts = np.vstack([by_id[i] for i in used]).astype(np.float64)
    rv, _ = cv2.Rodrigues(T_init[:3, :3])
    tv = T_init[:3, 3].reshape(3, 1).copy()  # must be 3x1: a flat (3,) tvec breaks RefineLM in OpenCV 5
    rv, tv = cv2.solvePnPRefineLM(obj, img_pts, intr.K_np, intr.dist_np, rv, tv)
    proj, _ = cv2.projectPoints(obj, rv, tv, intr.K_np, intr.dist_np)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean())
    return C.make_T(cv2.Rodrigues(rv)[0], tv.ravel()), used, err


def cube_pose_fused(views: list[tuple[np.ndarray, C.Intrinsics, np.ndarray]], layout: dict, det=None,
                    max_err: float = 2.0, depth_weight: float = 0.1):
    """T_world_cube from several cameras, fused per camera instead of in one PnP.

    Each camera gives a pose from its own tags (cube_pose_joint). Orientation: the average
    of the cameras' rotations. Position: weighted least squares in which each camera is
    trusted fully across its image and only `depth_weight` along its viewing ray (depth from
    a small tag's apparent size is its weakest direction). A single PnP over all views was
    tried first (removed): with one 4-corner tag per camera it bent the orientation by ~15 deg
    to absorb a ~10 mm disagreement between the cameras.
    Returns (T_world_cube or None, {view index: (ids, reprojection px, T_world_cube from that view)}).
    """
    det = det or detector()
    Rs, A, b, info = [], np.zeros((3, 3)), np.zeros(3), {}
    for vi, (img, intr, T_wc) in enumerate(views):
        T_cc, used, err = cube_pose_joint(img, intr, layout, det, max_err)
        if T_cc is None:
            continue
        T_w = T_wc @ T_cc
        ray = T_w[:3, 3] - T_wc[:3, 3]
        ray /= np.linalg.norm(ray)
        W = np.eye(3) - (1.0 - depth_weight) * np.outer(ray, ray)
        A += W
        b += W @ T_w[:3, 3]
        Rs.append(T_w[:3, :3])
        info[vi] = (used, err, T_w)
    if not Rs:
        return None, {}
    return C.make_T(C.average_rotations(Rs), np.linalg.solve(A, b)), info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--cam", default="top")
    c.add_argument("--every", type=float, default=0.4, help="min seconds between saved frames")
    s = sub.add_parser("solve")
    s.add_argument("--cam", default="top")
    s.add_argument("--max-err", type=float, default=1.5, help="drop tag detections with reprojection > px")
    args = ap.parse_args()
    {"capture": capture, "solve": solve}[args.cmd](args)


if __name__ == "__main__":
    main()
