# ORCA 手 real2sim 同步数据集

分支 `real2sim-cameras`。由 `real2sim/recording/package.py` 生成；详细过程见
`docs/real2sim_camera_calibration.md`（第 8 节）。

## 结构

```
calib/            相机位姿与内参、rig.yaml（全部测量值）、方块标签布局
<run>/dataset.csv       每个超声波采样一行（约 9–10 Hz）
<run>/dataset_twin.csv  同上 + 完整数字孪生（sim 手按回放指令、方块按相机姿态）
<run>/meta.json         录制信息（设备、分辨率、时钟原点）
<run>/twin_meta.json    回放在录制中的开始时刻
<run>/replay_log.csv    回放每帧发出的时刻（t = 录制时间基）
<run>/<replay>.yaml     回放的关节轨迹（orca_core 关节名，度，30 Hz）
```

图片（每个相机 1920x1080 JPEG，30 fps）不在包里，在 `real2sim/data/recordings/<run>/top|side/`；
CSV 里的 `top_file` / `side_file` 就是那里的相对路径。需要的话单独打包。

## 时间

所有时间 `t` 都是同一个主机时钟（`time.perf_counter`，单位 s，录制进程启动起算），
是数据**到达主机**的时刻。相机相对超声波的延迟已在对齐时扣除（top +9 ms，side −27 ms）。
`replay_log.csv` 的 `t` 与之同一时间基，可直接对齐。

## 坐标系

sim 世界坐标系（米）：手装在 (0, 0, 0.16)，手指指向 +x，掌心朝 +z，拇指在 −y 一侧，小指在 +y 一侧。
桌面 z = 0.092 m。方块 35 mm，sim 的"红色面" = 真实的**绿色面** = 方块 +z 面（AprilTag id 0）。

## 列

| 列 | 含义 |
|---|---|
| t | 时间 (s)，见上 |
| us_mm | 真实 HC-SR04 读数 (mm)，空 = 无回波 |
| top_file, side_file | 配对的相机帧（相对 recordings/<run>/） |
| top_dt_ms, side_dt_ms | 配对帧与超声波采样的时间差 (ms) |
| frames_ok | 两帧都在 25 ms 内 |
| cube_found | 相机识别到方块（AprilTag） |
| tags_top, tags_side | 各相机用到的标签 id |
| cube_x, cube_y, cube_z | 方块中心，世界坐标 (m) |
| cube_qw..cube_qz | 方块姿态四元数（世界坐标，wxyz） |
| green_up_deg | 绿色面法向与竖直向上的夹角 |
| face_incidence_deg | 超声波光束与正对它的方块面的夹角 |
| sim_nearest_mm | sim 超声波（只看方块，手为默认姿态）：15° 圆锥内最近命中 |
| sim_specular_mm | 同上，但只有入射角 ≤ 25° 的射线算回波 |
| real_sees_cube, sim_*_sees_cube | 读数 ≤ 150 mm |
| replay_frame | （twin）当时回放到的帧号 |
| sim_twin_mm | （twin）完整孪生的超声波读数 |
| sim_twin_hit | （twin）回波来源：cube / hand / bench / none |
| sim_twin_sees, real_sees_any | 150 mm 内是否有东西 |

精度：相机孪生约 5 mm；超声波静态 ±4 mm（static01：sim − 真实中位数 +0.7 mm，MAE 5.9 mm）；20 mm 以内是盲区。

## 注意

- 超声波传感器只是放在掌根上：手腕动过的回放（run03、run04、run06）超声波孪生无效。
- 关节映射：run08 用重新标定后的映射（`joint_map.py`：无名指/小指外展 sim = core + (−10° / −14°)）；
  run05–run07 的 `dataset_twin.csv` 是用当时的映射（符号 −1，−20° / −27°）算的。
- 回放是开环的，policy 看不到真实方块。

## 各次录制

| 录制 | 说明 | 行数 | 识别到方块 |
|---|---|---|---|
| angle01 | 入射角实验：方块由人手转动（人手入镜），无回放 | 556 | 553 |
| angle02 | 入射角实验，约 7 cm 处，方块由人手转动 | 567 | 535 |
| static01 | 方块静止放在掌心几个位置（拇指掰开），用于复核传感器位置；挪动之间人手入镜 | 291 | 269 |
| run01 | 第一次 policy 回放，手腕锁定，关节映射未修正；方块飞出 | 361 | 223 |
| run03 | 手腕 −10~+10°；回放开始时刻无法可靠确定，传感器随手腕移动 → 孪生无效 | 215 | 153 |
| run04 | 手腕转动 → 传感器移动，超声波孪生无效；带时间记录回放（中途停止） | 276 | 50 |
| run05 | 手腕锁定，绿色面朝小指；方块比训练位置偏向手腕 1.5–2 cm，手指没碰到方块 | 482 | 482 |
| run06 | 手腕 −15~+15°；传感器跟着手腕移动（相关 0.86/0.88），超声波孪生无效 | 483 | 483 |
| run07 | 手腕锁定，绿色面朝小指，方块在训练位置；约 7 s 从指尖一侧掉落。有效 | 459 | 89 |
| run08 | 重新标定后的新映射，手腕锁定，绿色面朝小指；方块转 124° 后从小指一侧掉落，人手在下方远处接住（未接触）。有效 | 553 | 127 |
