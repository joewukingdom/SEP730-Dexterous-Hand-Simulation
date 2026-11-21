# real2sim/data — raw and processed bench data / 实验台原始与处理后数据

Full method and results: `docs/real2sim_camera_calibration_en.md` (English) /
`docs/real2sim_camera_calibration.md` (中文). Ready-made CSV package for analysis:
`real2sim/dataset_package/` (+ `.zip`).

完整方法和结果见上面两份笔记；直接分析用的 CSV 包在 `real2sim/dataset_package/`。

## What is in git / 哪些在 git 里

Only this README. The folder (~23 GB, mostly 1920x1080 JPEG frames) stays on the bench PC; ask
for it if you need raw frames. The CSV datasets, metadata, replay logs and trajectories are in
`real2sim/dataset_package/`.

只有这个 README 在 git 里；整个文件夹（约 23 GB，主要是相机帧）在实验台电脑上，需要原始画面请联系。
CSV 数据、meta、回放记录和轨迹都在 `real2sim/dataset_package/`。

`real2sim/recording/make_videos.py` turns a recording's frame folders into `top.mp4` / `side.mp4`
(H.264 CRF 28, about 5 MB per minute). **Video frame N = file N** (`top/000142.jpg` = frame 142);
on run08 AprilTag corners from the video differ from the JPEG by a median 1 px. Read a frame:

```python
import cv2
cap = cv2.VideoCapture("real2sim/data/recordings/run08/top.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 142); ok, img = cap.read()   # = top/000142.jpg
img = cv2.undistort(img, K, dist)   # K, dist from real2sim/cameras/calib/top_intrinsics.json
```

## Folders / 文件夹

| Folder | Content / 内容 | Notes section |
|---|---|---|
| `intr_top/`, `intr_side/` | final intrinsic calibration photos (PNG, ChArUco 7x5, 35 mm) / 最终内参照片 | 2 |
| `intr/` | earlier intrinsic rounds 1–3 (the failed ones, JPEG copies) / 前几轮失败的内参照片 | 2.1 |
| `extr_top/`, `extr_side/` | extrinsic photos (board flat / board against the tower) / 外参照片 | 4, 5 |
| `extr_top_old/` | top extrinsics before the hand was moved / 手移动前的 top 外参 | 4.2 |
| `us_check/pos1..pos6/` | ultrasonic distance experiment: both camera images + real reading per cube position / 超声波距离实验 | 8.3 |
| `us_check/tagcheck/` | first two-camera tag consistency check / 两相机标签一致性检查 | 8.5 |
| `cube_tags/top/` | 407 frames of the hand-turned cube used to solve the tag layout (JPEG copies) / 求标签布局用的 407 帧 | 8.5 |
| `live/` | single live captures / 单次实时截图 | – |
| `policy/` | policy trajectories in the sim (intermediate) / sim 中的 policy 轨迹 | 8.6 |
| `replay_logs/<log>.csv` | time of every replay frame sent to the hand (`perf_counter`; subtract `perf_counter_origin` from the recording's `meta.json`) / 回放每帧的发送时刻 | 8.9 |
| `recordings/<run>/` | synchronized recordings, see below / 同步录制 | 8.4– |

### Files in a recording / 每次录制里的文件

| File | Content |
|---|---|
| `meta.json` | start time (UTC), devices, resolution, `perf_counter_origin`, dropped frames |
| `top_frames.csv`, `side_frames.csv` | `frame, t, file` for the frames **inside** the recording window (the folders/videos also hold frames from the countdown before it) |
| `ultrasonic.csv` | `sample, t, dist_mm, raw` (empty dist = no echo) |
| `aligned.csv` | each ultrasonic sample with its nearest top/side frame (`align.py`) |
| `dataset.csv` | + cube pose from the AprilTags and the sim ultrasonic (`process.py`) |
| `dataset_twin.csv`, `twin_meta.json`, `twin_overlay.jpg` | + full digital twin (`twin.py`) |
| `pose_check.jpg`, `pose_check_scores.json`, `zoom_*.jpg`, `abd_fit.json`, `joint_fit.json` | joint checks (`pose_check.py`, `finger_zoom.py`) |
| `snapshot/` | `top.png`, `side.png`, `real_reading.json`, `pose_check.jpg`: one synchronized still for the cube start pose |
| `top.mp4`, `side.mp4` | the camera frames (see above) |

All `t` are seconds on one host clock (`time.perf_counter` since the recorder started), the
time a sample arrived. 所有 `t` 都是同一个主机时钟。

### Recordings / 录制列表

| Run | What / 内容 | Valid for |
|---|---|---|
| `_test_sync`, `sync01`–`sync03` | recorder tests and the latency test (sync03: card flicked through the beam) / 录制测试与延迟测试 | sync03: latency |
| `tagcheck2` | tag check, synchronized / 标签检查 | cube pose |
| `angle01`, `angle02` | cube turned by hand ±45° on the palm (angle01 too close: dead zone) / 入射角实验 | angle02: incidence threshold |
| `start01`–`start08` | 3 s stills of the cube start pose before each policy run / 每次 policy 实验前的起始姿态 | start pose |
| `run01`, `run02` | first policy replays (old joint mapping; run02 has no dataset) | cameras, cube |
| `run03`, `run04`, `run06` | policy replays with the wrist moving → the loose sensor moved: ultrasonic twin invalid | cameras, cube |
| `run05` | wrist locked, cube off the training position, fingers did not reach it | ultrasonic, cube |
| `run07`, `run08` | wrist locked, cube at the training position; run08 with the current mapping (cube turned 124°) | **full twin** |
| `static01` | cube still at several palm positions, thumb opened by hand | ultrasonic static check |
| `posecheck01`–`posecheck03` | 6 known hand poses (before / after the first recalibration) | joint check |
| `abdcheck01`–`abdcheck03` | abduction joints one at a time; replay start guessed from image motion (no replay log) | superseded by fullcheck01 |
| `fullcheck01` | every finger joint alone at two angles, aligned exactly by the replay log | joint check, current mapping |

Hand calibration history / 手的标定历史: `calibration.yaml` from 2026-09-08 → recalibrated at
2026-11-20 19:14 (before posecheck03); not changed after that. The current `joint_map.py` is fitted
on fullcheck01.
