"""Ultrasonic digital twin: locate the real HC-SR04 with the two calibrated cameras, and
compare the sim sensor (cone of rangefinders in the _real2sim scene) with the real one.

    # 1. where is the sensor? (pixels of the transducer front faces; prints the rig.yaml block)
    python real2sim/ultrasonic/twin_check.py locate --top-a 1045 797 --side-a 1338 515 --top-b 1155 782

    # 2. sim vs real: put the sim cube where the real one is, read both sensors
    python real2sim/ultrasonic/twin_check.py compare --top-img real2sim/data/live/top.png \
        --side-top-edge 1180 492 --cube-mm 35 --real-port COM4

`locate`: transducer `a` is seen by both cameras and triangulated; `b` only by the top
camera and is put at a's height. Pixels are raw (distorted) image coordinates.
`compare`: the real cube's pose comes from the cameras: its green top face is found in the
top image (centre + orientation, IPPE on the face outline), its height from the side image
(the row of the cube's top edge, `--side-top-edge u v`; the side camera is almost level, so
it measures height far better than the top camera does). The cube is then placed in the sim
and the sim reading is printed next to the real one (read-only from the serial port).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cameras"))
sys.path.insert(0, str(HERE.parents[1] / "hardware" / "ultrasonic"))
import camlib as C  # noqa: E402


def cameras() -> dict:
    cal = json.loads((C.CALIB_DIR / "cameras.json").read_text())
    return {n: (np.asarray(c["T_world_cam_opencv"]), C.Intrinsics.load(C.intrinsics_path(n)))
            for n, c in cal["cameras"].items()}


def ray(cams: dict, cam: str, uv, undistorted: bool = False) -> tuple[np.ndarray, np.ndarray]:
    T, intr = cams[cam]
    if undistorted:
        n = np.linalg.inv(intr.K_np) @ np.array([uv[0], uv[1], 1.0])
    else:
        n2 = cv2.undistortPoints(np.float64([[uv]]), intr.K_np, intr.dist_np).reshape(2)
        n = np.array([n2[0], n2[1], 1.0])
    d = T[:3, :3] @ n
    return T[:3, 3], d / np.linalg.norm(d)


def triangulate(r1, r2) -> tuple[np.ndarray, float]:
    (o1, d1), (o2, d2) = r1, r2
    A = np.array([[d1 @ d1, -d1 @ d2], [d1 @ d2, -d2 @ d2]])
    s, t = np.linalg.solve(A, np.array([(o2 - o1) @ d1, (o2 - o1) @ d2]))
    p1, p2 = o1 + s * d1, o2 + t * d2
    return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2))


def on_plane_z(r, z: float) -> np.ndarray:
    o, d = r
    return o + (z - o[2]) / d[2] * d


def locate(args) -> None:
    cams = cameras()
    pa, gap = triangulate(ray(cams, "top", args.top_a), ray(cams, "side", args.side_a))
    pb = on_plane_z(ray(cams, "top", args.top_b), pa[2])
    pos = (pa + pb) / 2
    lat = pb - pa
    lat[2] = 0.0
    beam = np.cross(lat, [0.0, 0.0, 1.0])
    beam /= np.linalg.norm(beam)
    if beam[0] < 0:  # towards the fingers (+x)
        beam = -beam
    print(f"transducer a {np.round(pa, 4)} (ray gap {gap * 1000:.1f} mm), b {np.round(pb, 4)}, "
          f"spacing {np.linalg.norm(pb - pa) * 1000:.1f} mm (HC-SR04: 26 mm)")
    print("rig.yaml:\nultrasonic:\n"
          f"  pos_m: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n"
          f"  dir: [{beam[0]:.3f}, {beam[1]:.3f}, {beam[2]:.1f}]\n"
          "  half_angle_deg: 15.0\n  rings: 3\n  n_az: 8")


def cube_pose(cams: dict, top_img: np.ndarray, side_top_edge, edge_m: float):
    """Real cube centre + rotation from the green top face (top cam) and its height (side cam)."""
    T, intr = cams["top"]
    und = cv2.undistort(top_img, intr.K_np, intr.dist_np)
    mask = cv2.inRange(cv2.cvtColor(und, cv2.COLOR_BGR2HSV), (40, 60, 80), (85, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    quad = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True).reshape(-1, 2).astype(np.float64)
    if len(quad) != 4:
        raise SystemExit(f"green top face: expected 4 corners, found {len(quad)}")
    c0 = quad.mean(0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - c0[1], quad[:, 0] - c0[0]))]
    a = edge_m / 2
    obj = np.array([[-a, a, 0], [a, a, 0], [a, -a, 0], [-a, -a, 0]], float)
    best = None
    for k in range(4):
        q = np.roll(quad, k, axis=0)
        _, rv, tv = cv2.solvePnP(obj, q, intr.K_np, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        proj, _ = cv2.projectPoints(obj, rv, tv, intr.K_np, None)
        e = float(np.linalg.norm(proj.reshape(-1, 2) - q, axis=1).mean())
        if best is None or e < best[0]:
            best = (e, rv, tv)
    R = T[:3, :3] @ cv2.Rodrigues(best[1])[0]
    n = R[:, 2] if R[2, 2] > 0 else -R[:, 2]
    x = R[:, 0] - (R[:, 0] @ n) * n
    x /= np.linalg.norm(x)
    Rw = np.column_stack([x, np.cross(n, x), n])
    r_c = ray(cams, "top", c0, undistorted=True)
    # top-face height from the side camera: the near (+y) top edge seen at side_top_edge
    z = 0.2
    o, d = ray(cams, "side", side_top_edge)
    for _ in range(20):
        y_edge = on_plane_z(r_c, z)[1] + a
        z = float((o + (y_edge - o[1]) / d[1] * d)[2])
    face = on_plane_z(r_c, z)
    return face - a * n, Rw, {"top_face_z": z, "tilt_deg": float(np.degrees(np.arccos(n[2]))), "pnp_px": best[0]}


def read_real(port: str, seconds: float) -> list[float | None]:
    import serial

    import read_ultrasonic as R
    out = []
    with serial.Serial(port, 115200, timeout=1.0) as ser:
        time.sleep(0.3)
        ser.reset_input_buffer()
        t0 = time.time()
        for v in R.readings(ser):
            out.append(None if v is None else v / 1000.0)
            if time.time() - t0 > seconds:
                break
    return out


def compare(args) -> None:
    cams = cameras()
    rig = C.load_rig(C.HERE / "rig.yaml")
    centre, Rw, info = cube_pose(cams, cv2.imread(str(args.top_img)), args.side_top_edge, args.cube_mm / 1000)
    print(f"real cube: centre {np.round(centre, 4)}, top face z {info['top_face_z']:.4f}, "
          f"tilt {info['tilt_deg']:.1f} deg (face fit {info['pnp_px']:.2f} px)")
    scene = (C.SCENES_DIR / rig["scene"].replace(".xml", "_real2sim.xml")).resolve()
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    j = m.joint("cube_freejoint").qposadr[0]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rw.flatten())
    sim_edge = 2 * m.geom("task_cube_body").size[0]
    # keep the real cube's near faces where they are: the sim cube is a little bigger
    d.qpos[j:j + 3] = centre
    d.qpos[j + 3:j + 7] = q
    mujoco.mj_forward(m, d)
    sim = C.ultrasonic_reading(m, d)
    rays = [m.sensor(i).name for i in range(m.nsensor) if m.sensor(i).name.startswith("us_")]
    print(f"sim sensor ({len(rays)} rays, {rig['ultrasonic']['half_angle_deg']} deg cone): "
          f"{'no echo' if sim is None else f'{sim * 1000:.1f} mm'}  "
          f"(sim cube edge {sim_edge * 1000:.0f} mm vs real {args.cube_mm:.0f} mm)")
    if args.real_port:
        vals = read_real(args.real_port, args.seconds)
        hits = [v for v in vals if v is not None]
        if hits:
            real = statistics.median(hits)
            print(f"real sensor ({args.real_port}, {len(vals)} readings): median {real * 1000:.1f} mm, "
                  f"min {min(hits) * 1000:.1f}, max {max(hits) * 1000:.1f}")
            if sim is not None:
                print(f"sim - real = {(sim - real) * 1000:+.1f} mm")
        else:
            print("real sensor: no echo")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lo = sub.add_parser("locate")
    lo.add_argument("--top-a", type=float, nargs=2, required=True)
    lo.add_argument("--side-a", type=float, nargs=2, required=True)
    lo.add_argument("--top-b", type=float, nargs=2, required=True)
    co = sub.add_parser("compare")
    co.add_argument("--top-img", type=Path, default=C.DATA_DIR / "live" / "top.png")
    co.add_argument("--side-top-edge", type=float, nargs=2, required=True)
    co.add_argument("--cube-mm", type=float, default=35.0)
    co.add_argument("--real-port")
    co.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    locate(args) if args.cmd == "locate" else compare(args)


if __name__ == "__main__":
    main()
