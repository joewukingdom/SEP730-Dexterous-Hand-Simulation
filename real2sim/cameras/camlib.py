"""Shared helpers for the camera real2sim pipeline.

Conventions
-----------
* Units: metres and radians internally; the rig file uses mm / degrees where noted.
* OpenCV camera frame: x right, y down, z forward (looking direction).
* MuJoCo camera frame: x right, y up, looks along -z.  R_mj = R_cv @ diag(1, -1, -1).
* ChArUco board frame as OpenCV defines it ("board_cv"): origin at the outer corner
  of the first square, x along the squares_x edge, y along the squares_y edge, z
  pointing INTO the board (away from a camera that sees its printed side).
* "board_in_world" in rig.yaml is given for the board CENTRE: start with the board
  flat, printed side up, long (squares_x) edge along world x; tilt it by `tilt_deg`
  about its long edge (90 = standing up with the print facing world -y, -90 = facing
  +y), then turn it by `yaw_deg` about world z, then move the centre to `pos_m`.
  In that "centre" frame z is the print's normal; rotating by 180 deg about x gives
  board_cv (shifted to the corner).
* Each camera may use its own board placement (cameras.<name>.board_in_world in
  rig.yaml); the top-level board_in_world is the default.
* MuJoCo intrinsics: focalpixel = (fx, fy),
  principalpixel = (W/2 - (cx + 0.5), H/2 - (cy + 0.5)). The +0.5 is because OpenCV puts
  pixel centres at integer coordinates and OpenGL at +0.5. Verified by rendering
  (tests/test_real2sim_cameras.py): sim-detected corners land on the K-projection
  to < 0.1 px.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SCENES_DIR = REPO_ROOT / "src" / "orca_sim" / "scenes" / "v2"
CALIB_DIR = HERE / "calib"
DATA_DIR = REPO_ROOT / "real2sim" / "data"
BOARD_TEXTURE_NAME = "real2sim_board.png"
BOARD_TEXTURE_PX_PER_MM = 8
SENSOR_PIXEL_M = 3.0e-6  # arbitrary; MuJoCo only uses it to convert focalpixel <-> focal


# --------------------------------------------------------------------------- rig / board


def load_rig(path: Path | str = HERE / "rig.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def make_board(board_cfg: dict) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board_cfg["dictionary"]))
    return cv2.aruco.CharucoBoard(
        (int(board_cfg["squares_x"]), int(board_cfg["squares_y"])),
        board_cfg["square_mm"] / 1000.0,
        board_cfg["marker_mm"] / 1000.0,
        dictionary,
    )


def board_size_m(board_cfg: dict) -> tuple[float, float]:
    s = board_cfg["square_mm"] / 1000.0
    return board_cfg["squares_x"] * s, board_cfg["squares_y"] * s


def board_image(board_cfg: dict, px_per_mm: float) -> np.ndarray:
    """The board area only (no margin), exactly squares * square_mm in size."""
    board = make_board(board_cfg)
    w = int(round(board_cfg["squares_x"] * board_cfg["square_mm"] * px_per_mm))
    h = int(round(board_cfg["squares_y"] * board_cfg["square_mm"] * px_per_mm))
    return board.generateImage((w, h), marginSize=0, borderBits=1)


class BoardDetector:
    def __init__(self, board_cfg: dict):
        self.board = make_board(board_cfg)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.CharucoDetector(self.board, cv2.aruco.CharucoParameters(), params)

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (corners Nx2 float32, ids N int); empty arrays if nothing found."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0,), np.int32)
        return corners.reshape(-1, 2).astype(np.float32), ids.reshape(-1).astype(np.int32)

    def object_points(self, ids: np.ndarray) -> np.ndarray:
        """3D corner positions (board_cv frame, metres) for the given charuco ids."""
        return np.asarray(self.board.getChessboardCorners(), np.float64)[ids]


# --------------------------------------------------------------------------- transforms


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def T_world_boardcv(board_cfg: dict, board_in_world: dict) -> np.ndarray:
    """Pose of the OpenCV board frame in the MuJoCo world frame."""
    lx, ly = board_size_m(board_cfg)
    R = rot_z(math.radians(board_in_world.get("yaw_deg", 0.0))) @ rot_x(math.radians(board_in_world.get("tilt_deg", 0.0)))
    T_world_centre = make_T(R, np.asarray(board_in_world["pos_m"], float))
    # centre frame (z up)  ->  board_cv (z down, origin at the first corner)
    T_centre_cv = make_T(rot_x(math.pi), np.array([-lx / 2, ly / 2, 0.0]))
    return T_world_centre @ T_centre_cv


def board_pose_for(rig: dict, cam: str) -> dict:
    """Board placement used for `cam`'s extrinsic capture (per-camera override or default)."""
    return rig["cameras"][cam].get("board_in_world") or rig["board_in_world"]


def average_rotations(Rs: list[np.ndarray]) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(np.clip(c, -1.0, 1.0)))


# --------------------------------------------------------------------------- intrinsics


@dataclass
class Intrinsics:
    width: int
    height: int
    K: list  # 3x3
    dist: list  # OpenCV distortion coefficients
    rms_px: float = float("nan")
    n_images: int = 0

    @property
    def K_np(self) -> np.ndarray:
        return np.asarray(self.K, float)

    @property
    def dist_np(self) -> np.ndarray:
        return np.asarray(self.dist, float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Intrinsics":
        return cls(**json.loads(Path(path).read_text()))


def intrinsics_path(cam: str) -> Path:
    return CALIB_DIR / f"{cam}_intrinsics.json"


# --------------------------------------------------------------------------- MuJoCo camera


@dataclass
class MjCamera:
    name: str
    T_world_camcv: np.ndarray  # 4x4
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_intrinsics(cls, name: str, T_world_camcv: np.ndarray, intr: Intrinsics) -> "MjCamera":
        K = intr.K_np
        return cls(name, T_world_camcv, intr.width, intr.height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def xml(self) -> str:
        R_mj = self.T_world_camcv[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        pos = self.T_world_camcv[:3, 3]
        xyaxes = np.concatenate([R_mj[:, 0], R_mj[:, 1]])
        f = lambda v: " ".join(f"{x:.6f}" for x in v)  # noqa: E731
        W, H = self.width, self.height
        return (
            f'<camera name="{self.name}" pos="{f(pos)}" xyaxes="{f(xyaxes)}" '
            f'resolution="{W} {H}" sensorsize="{W * SENSOR_PIXEL_M:.8f} {H * SENSOR_PIXEL_M:.8f}" '
            f'focalpixel="{self.fx:.3f} {self.fy:.3f}" '
            f'principalpixel="{W / 2 - (self.cx + 0.5):.3f} {H / 2 - (self.cy + 0.5):.3f}"/>'
        )


# --------------------------------------------------------------------------- scene


def _board_mesh_xml(board_cfg: dict, thickness: float = 0.001) -> str:
    """Thin box in the board_cv frame; the printed (textured) side faces -z_cv."""
    lx, ly = board_size_m(board_cfg)
    t = thickness
    xy = [(0, 0), (lx, 0), (lx, ly), (0, ly)]
    verts = [(x, y, 0.0) for x, y in xy] + [(x, y, t) for x, y in xy]
    # texture row 0 is the top of the generated board image (board_cv y = 0);
    # MuJoCo puts v = 0 at the image's first row.
    uv = [(x / lx, y / ly) for x, y in xy] * 2
    faces = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    fmt = lambda rows: "  ".join(" ".join(f"{v:.6f}" for v in r) for r in rows)  # noqa: E731
    return (
        # scale="1 1 1": the hand's options.xml sets a default mesh scale of 0.001 (mm STLs)
        f'<mesh name="real2sim_board" scale="1 1 1" vertex="{fmt(verts)}" texcoord="{fmt(uv)}" '
        f'face="{"  ".join(" ".join(map(str, f)) for f in faces)}"/>'
    )


def _quat_xml(R: np.ndarray) -> str:
    q = np.zeros(4)
    import mujoco

    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return " ".join(f"{v:.8f}" for v in q)


def write_board_texture(board_cfg: dict, name: str = BOARD_TEXTURE_NAME) -> Path:
    path = SCENES_DIR / name
    cv2.imwrite(str(path), board_image(board_cfg, BOARD_TEXTURE_PX_PER_MM))
    return path


def build_scene_xml(
    base_scene: str,
    board_cfg: dict,
    T_world_board: np.ndarray,
    cameras: list[MjCamera],
    *,
    board_mocap: bool = False,
    model_name: str = "real2sim",
    texture_file: str = BOARD_TEXTURE_NAME,
    extra_boards: list[np.ndarray] = (),
    bench: dict | None = None,
) -> str:
    """Wrapper MJCF that includes `base_scene` (same directory) and adds the calibration
    board (visual only, geom group 3 so normal renders hide it) and the cameras.
    `extra_boards`: more static copies of the board (real2sim_board_1, ...), for previews.
    `bench`: {"top_z": m, optional "center_xy", "half_size_xy", "rgba"} adds the real bench
    as a solid box whose top is at top_z. The base scene's floor (z = 0) stays below it, so
    the training scenes are unchanged."""
    bench_xml = ""
    if bench:
        cx, cy = bench.get("center_xy", [0.1, 0.0])
        hx, hy = bench.get("half_size_xy", [0.6, 0.6])
        rgba = " ".join(str(v) for v in bench.get("rgba", [0.86, 0.86, 0.84, 1]))
        top = float(bench["top_z"])
        bench_xml = (f'\n    <geom name="real2sim_bench" type="box" pos="{cx} {cy} {top - 0.01:.4f}" '
                     f'size="{hx} {hy} 0.01" rgba="{rgba}" friction="1 0.005 0.0001"/>')
    max_w = max([c.width for c in cameras] + [640])
    max_h = max([c.height for c in cameras] + [480])
    cams = "\n    ".join(c.xml() for c in cameras)
    mocap = ' mocap="true"' if board_mocap else ""
    extra = "".join(
        f'''
    <body name="real2sim_board_{i}" pos="{' '.join(f'{v:.6f}' for v in T[:3, 3])}" quat="{_quat_xml(T[:3, :3])}">
      <geom name="real2sim_board_{i}" type="mesh" mesh="real2sim_board" material="real2sim_board"
            contype="0" conaffinity="0" group="3" mass="0.001"/>
    </body>'''
        for i, T in enumerate(extra_boards, start=1)
    )
    return f"""<?xml version="1.0" ?>
<!-- GENERATED by real2sim/cameras -- do not edit by hand, re-run calibrate_extrinsics.py -->
<mujoco model="{model_name}">
  <include file="{base_scene}"/>
  <visual>
    <global offwidth="{max_w}" offheight="{max_h}"/>
  </visual>
  <asset>
    <texture name="real2sim_board" type="2d" file="{texture_file}"/>
    <material name="real2sim_board" texture="real2sim_board" specular="0" shininess="0"/>
    {_board_mesh_xml(board_cfg)}
  </asset>
  <worldbody>
    <body name="real2sim_board" pos="{' '.join(f'{v:.6f}' for v in T_world_board[:3, 3])}" quat="{_quat_xml(T_world_board[:3, :3])}"{mocap}>
      <geom name="real2sim_board" type="mesh" mesh="real2sim_board" material="real2sim_board"
            contype="0" conaffinity="0" group="3" mass="0.001"/>
    </body>{extra}{bench_xml}
    {cams}
  </worldbody>
</mujoco>
"""


def render_camera(model, data, cam_name: str, *, show_board: bool = True) -> np.ndarray:
    """BGR render at the camera's calibrated resolution."""
    import mujoco

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    W, H = (int(v) for v in model.cam_resolution[cid])
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1 if show_board else 0
    with mujoco.Renderer(model, H, W) as r:
        r.update_scene(data, camera=cam_name, scene_option=opt)
        rgb = r.render()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- capture


def open_camera(device: int, width: int, height: int) -> cv2.VideoCapture:
    """Open a UVC camera. On Windows MSMF is tried first: DirectShow can be hijacked by
    virtual cameras (e.g. NVIDIA Broadcast)."""
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if os.name == "nt" else [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(device, be)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if ok and frame is not None:
            if (frame.shape[1], frame.shape[0]) != (width, height):
                print(f"  camera {device}: asked for {width}x{height}, got {frame.shape[1]}x{frame.shape[0]}")
            return cap
        cap.release()
    raise RuntimeError(f"could not open camera device {device}")
