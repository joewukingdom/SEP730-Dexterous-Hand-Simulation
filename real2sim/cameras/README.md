# Camera real2sim

Calibrate the two real cameras (Brio 100, `top` and `side`) and put identical cameras
into the MuJoCo scene, so that a rendered frame lines up with the real frame.

Output: `src/orca_sim/scenes/v2/scene_right_cube_orientation_real2sim.xml` (the cube scene plus
cameras `top` and `side` with the real intrinsics and poses) and `calib/cameras.json`.

Dependencies: `pip install opencv-contrib-python pyyaml pillow` (plus `mujoco`, `numpy`).
Run everything from the repository root.

## 1. Print the targets

```bash
python real2sim/cameras/make_targets.py
```

- `targets/board_letter.pdf` (US Letter; `--paper a4` for A4): print at **100% / actual size**. The bar under the board must
  measure 100 mm. Glue it onto something flat and rigid (foam board, acrylic, a clipboard).
  Measure one square with calipers and write the value into `rig.yaml` (`square_mm`). Scale
  `marker_mm` by the same factor. A 1% scale error becomes a 1% depth error.
- `targets/cube_tags.pdf`: AprilTags for the cube faces. `cube_tags.json` lists which tag
  goes on which face. Not needed for the camera calibration itself.

## 2. Intrinsics (focal length, principal point, lens distortion), once per camera

```bash
python real2sim/cameras/capture.py --list            # check device indices, fix rig.yaml if needed
python real2sim/cameras/capture.py intr               # both cameras side by side
python real2sim/cameras/calibrate_intrinsics.py intr --cam top
python real2sim/cameras/calibrate_intrinsics.py intr --cam side
```

Each camera is calibrated on its own (intrinsics belong to one camera), but you can
capture both in one session. Every SPACE saves a frame from both cameras, and frames
where a camera does not see the board are skipped for that camera. The board rarely
fills the corners of both views at once, so work through one camera's image area and
then the other's, or use `--cams top` to show just one.

Hold the board in front of the camera and take 25-40 frames (SPACE, or A for auto-save
every 1.5 s). Cover the **whole image, including the corners and edges**, tilt the board
up to about 45° in every direction, and vary the distance. Keep it still while a frame is
saved, because motion blur ruins corners. Only save frames where the preview counter is green.

Good result: RMS < 0.5 px, every cell of the printed coverage grid non-zero.
Brio 100 is fixed-focus. Once its intrinsics are calibrated they stay valid as long as the
resolution stays 1920x1080, even if the camera is moved.

## 3. Extrinsics (camera pose in the sim world)

Each camera is tied to the sim world through its own board placement, so the two cameras
do not need to see the board at the same time:

| camera | board placement (rig.yaml) | why |
|---|---|---|
| `top` | flat on the bench next to the hand (`board_in_world`) | looks down, sees a flat board square-on |
| `side` | standing up between hand and camera, print facing the camera (`cameras.side.board_in_world`, `tilt_deg: -90`) | it is almost level with the bench, and a flat board would be too foreshortened |

1. Measure **H**, the height of the bottom of the black forearm tower above the bench. In
   the sim that bottom is at z = 0.112, so the bench is at z = 0.112 - H. Set the z values
   accordingly (see the comments in rig.yaml). Measure each board centre's x / y from
   the tower (front face at x = 0.099, centre line y = -0.010; fingers point +x, and the side
   camera is on the +y side).
2. `python real2sim/cameras/show_scene.py` shows both placements next to the sim hand.
   The real boards should sit in the same place relative to the real hand.
3. Capture each placement (keep the board perfectly still) and calibrate:
   ```bash
   python real2sim/cameras/capture.py extr_top  --cams top     # board flat
   python real2sim/cameras/capture.py extr_side --cams side    # board standing (use a square/box to keep it vertical)
   python real2sim/cameras/calibrate_extrinsics.py --cam-session top=extr_top --cam-session side=extr_side
   ```
   This writes the `_real2sim.xml` scene. Expect reproj < 0.5 px and spread < 1 mm.

Each camera's pose is only as good as its board measurement: a 5 mm placement error moves
that camera by 5 mm. If a flat placement works for both cameras, a single session also works:
`capture.py extr` + `calibrate_extrinsics.py extr`.

## 4. Check real vs sim

```bash
python real2sim/cameras/overlay.py --calib                         # the calibration frames (sanity)
python real2sim/cameras/capture.py val_side --cams side           # board moved to a NEW measured spot
python real2sim/cameras/overlay.py val_side --cams side --board-pos X Y Z --board-yaw DEG --board-tilt -90
python real2sim/cameras/show_scene.py --calibrated                # look through the sim cameras
```

Do the same for `top` (flat, `--board-tilt 0`). `overlay.py` detects the board in the
undistorted real frame and in the MuJoCo render and compares the corners by id. It saves
`calib/overlay_<cam>.png` and `calib/overlay_<session>.json`. Report the held-out numbers.
The real camera has lens distortion and MuJoCo renders a pinhole image, so real frames must
be undistorted with `cv2.undistort(img, K, dist)` (K and dist from `calib/cameras.json`)
before they are compared with sim frames or fed to a policy.

## What was actually done on the bench

`docs/real2sim_camera_calibration.md` records the calibration of 2026-11-18..20 step by step,
with figures, numbers, failed attempts and the error budget. Two things from it differ from
the generic steps above:

- `side`: the board stood on the bench with its back pressed against the tower's pinky-side
  face (so it is vertical and parallel to the tower without any alignment), and only two
  short distances were measured. See `rig.yaml`.
- `rig.yaml` `bench`: the generated `_real2sim` scene gets a solid bench at the real height
  (top at z = 0.092). The training scenes are unchanged. A cube dropped in the `_real2sim`
  scene rests at z ≈ 0.110, so a task env on it needs `drop_height` ≈ 0.13.

Other tools:

- `correct_bump.py CAM BEFORE.png AFTER.png`: a camera that was knocked slightly after its
  extrinsic capture is corrected from background feature matches (pure-rotation model). It
  refuses if the residual says the camera also moved.
- `calibrate_extrinsics.py --via side=top:SESSION`: calibrate a camera from frames in which an
  already calibrated camera sees the same board (no tape measure). Not used on the bench,
  because both cameras could not see the board fully at the same time.
- `make_figures.py`: regenerates the coverage and final overlay figures of the write-up.

## Test

`tests/test_real2sim_cameras.py` runs the whole pipeline on synthetic "real" images (MuJoCo
cameras with known intrinsics, distortion and poses; top camera with a flat board, low side
camera with a standing board, separate sessions). It recovers the focal length to within
0.3%, the camera positions to within about 3 mm and 0.3°, and real-vs-sim corners to within 0.1 px.
