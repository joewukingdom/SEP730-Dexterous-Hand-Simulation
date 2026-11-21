# ORCA hand real2sim synchronized dataset

Branch `real2sim-cameras`. Built by `real2sim/recording/package.py`; full method in
`docs/real2sim_camera_calibration_en.md` (Section 8). Chinese README: `README.md`.

## Layout

```
calib/                  camera poses and intrinsics, rig.yaml (all bench measurements), cube tag layout
<run>/dataset.csv       one row per ultrasonic sample (~9-10 Hz)
<run>/dataset_twin.csv  same + full digital twin (sim hand at the commanded angles, cube at the camera pose)
<run>/meta.json         recording info (devices, resolution, clock origin)
<run>/twin_meta.json    replay start within the recording
<run>/replay_log.csv    time each replay frame was sent (t = recording time base)
<run>/<replay>.yaml     replayed joint trajectory (orca_core joint names, degrees, 30 Hz)
```

Camera frames (1920x1080 JPEG, 30 fps per camera) are not in the package; they are in
`real2sim/data/recordings/<run>/top|side/`, and `top_file` / `side_file` in the CSVs are paths
relative to that folder. Ask if you need them packaged.

## Time

Every `t` is the same host clock (`time.perf_counter`, seconds since the recorder process
started): the time a sample **arrived on the host**. Camera latencies relative to the ultrasonic
are removed in the alignment (top +9 ms, side -27 ms). `replay_log.csv` uses the same time base.

## Frames

Sim world frame (m): hand mounted at (0, 0, 0.16), fingers along +x, palm facing +z, thumb on
the -y side, pinky on the +y side. Bench top at z = 0.092 m. Cube 35 mm; the sim's "red face" =
the real **green face** = cube +z face (AprilTag id 0).

## Columns

| Column | Meaning |
|---|---|
| t | time (s), see above |
| us_mm | real HC-SR04 reading (mm), empty = no echo |
| top_file, side_file | paired camera frames (relative to recordings/<run>/) |
| top_dt_ms, side_dt_ms | time between the paired frame and the ultrasonic sample (ms) |
| frames_ok | both frames within 25 ms |
| cube_found | cube detected by the cameras (AprilTags) |
| tags_top, tags_side | tag ids used per camera |
| cube_x, cube_y, cube_z | cube centre, world frame (m) |
| cube_qw..cube_qz | cube orientation quaternion (world, wxyz) |
| green_up_deg | angle between the green-face normal and world up |
| face_incidence_deg | angle between the ultrasonic beam and the cube face facing it |
| sim_nearest_mm | sim ultrasonic (cube only, hand in its default pose): nearest hit in the 15 deg cone |
| sim_specular_mm | same, but a ray only echoes at incidence <= 25 deg |
| real_sees_cube, sim_*_sees_cube | reading <= 150 mm |
| replay_frame | (twin) replay frame being executed |
| sim_twin_mm | (twin) full-twin ultrasonic reading |
| sim_twin_hit | (twin) echo source: cube / hand / bench / none |
| sim_twin_sees, real_sees_any | something within 150 mm |

Accuracy: camera twin ~5 mm; ultrasonic static +-4 mm (static01: sim - real median +0.7 mm,
MAE 5.9 mm); below 20 mm is the sensor's dead zone.

## Caveats

- The ultrasonic sensor only rests on the palm base: replays that moved the wrist (run03, run04,
  run06) have an invalid ultrasonic twin.
- Joint mapping: run08 uses the mapping measured on fullcheck01 (`joint_map.py`: ring/pinky
  abduction sim = core + (-10 / -14 deg)); the replays and `dataset_twin.csv` of run03-run07 used the
  abdcheck03 mapping (sign -1, -20 / -27 deg), whose ring/pinky direction is probably wrong.
- Replays are open loop; the policy does not see the real cube.

## Runs

| Run | Notes | Rows | Cube found |
|---|---|---|---|
| angle01 | incidence experiment: cube turned by hand (hand in view), no replay | 556 | 553 |
| angle02 | incidence experiment at ~7 cm, cube turned by hand | 567 | 535 |
| static01 | cube still at several palm positions (thumb opened), sensor position check; hand in view while moving it | 291 | 269 |
| run01 | first policy replay, wrist locked, joint mapping not yet corrected; cube flicked off | 361 | 223 |
| run03 | wrist -10..+10 deg; replay start not reliably known, sensor moves with the wrist -> twin invalid | 215 | 153 |
| run04 | wrist moved -> sensor moved, ultrasonic twin invalid; time-logged replay (stopped early) | 276 | 50 |
| run05 | wrist locked, green face to the pinky; cube 1.5-2 cm towards the wrist from the training position, fingers did not reach it | 482 | 482 |
| run06 | wrist -15..+15 deg; sensor follows the wrist (corr 0.86/0.88), ultrasonic twin invalid | 483 | 483 |
| run07 | wrist locked, green face to the pinky, cube at the training position; falls off the fingertip side at ~7 s. Valid | 459 | 89 |
| run08 | mapping from fullcheck01, wrist locked, green face to the pinky; cube turned 124 deg, then rolls off the pinky side, caught far below by a human hand (no contact). Valid | 553 | 127 |
