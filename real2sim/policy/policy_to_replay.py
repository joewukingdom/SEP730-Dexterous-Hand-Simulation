"""Run the trained PPO policy in the sim and turn its finger motion into a joint trajectory
that the real hand replays with orca_core's own tool (open loop).

    python real2sim/policy/policy_to_replay.py --start-from real2sim/data/recordings/start01/snapshot \
        --out policy_run01.yaml
    # then, in the orca folder (the usual tool, Ctrl+C stops it):
    uv run python scripts/replay_continuous.py orca_core/models/v2/orcahand_right/config.yaml \
        --replay-file policy_run01.yaml

--start-from: the sim cube starts where the cameras see the real cube (AprilTags), so the
first moves are the ones the policy would make for that cube. Without it, --init random/down.

Safety (the trajectory is only ever played by the user's replay tool, nothing here talks to
the hand):
  * wrist frozen: its action is forced to 0 in the sim and its angle is the config's neutral
    value in the file (moving the wrist shifts the whole hand and the sensor),
  * slowed down: time stretched by --slow, and more where needed so that no joint moves
    faster than --max-speed deg/s,
  * every angle clamped into the config's joint_roms,
  * --ramp seconds from the neutral pose into the first pose; at the end the thumb is
    released first and then the other fingers (--release seconds each, eased), because
    sending every finger back at once flicked the cube off the hand in run01.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORCA = Path(r"C:\Users\Owner\Desktop\orca")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parent / "cameras"))
sys.path.insert(0, str(HERE.parent / "cube"))
sys.path.insert(0, str(ORCA))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from orca_sim import OrcaHandRightCubeOrientation  # noqa: E402
from orca_sim.wrappers import RelativeJointTargetAction  # noqa: E402
from sim_to_real import JointBridge  # noqa: E402  (orca folder: sim <-> orca_core joint names/signs)

sys.path.insert(0, str(HERE.parent / "hand"))
from joint_map import read_core_degrees  # noqa: E402  (+ ring/pinky abd corrections measured on the bench)

CONFIG = ORCA / "orca_core" / "models" / "v2" / "orcahand_right" / "config.yaml"
MODEL = REPO / "models" / "ppo_reorient_relative"


def cube_from_snapshot(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    import camlib as C
    import cube_tags as CT
    layout = json.loads(CT.LAYOUT.read_text())
    cal = json.loads((C.CALIB_DIR / "cameras.json").read_text())
    views = [(cv2.imread(str(folder / f"{c}.png")), C.Intrinsics.load(C.intrinsics_path(c)),
              np.asarray(cal["cameras"][c]["T_world_cam_opencv"])) for c in ("top", "side")]
    T, info = CT.cube_pose_fused(views, layout)
    if T is None:
        raise SystemExit("no tag visible in the snapshot")
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, T[:3, :3].flatten())
    return T[:3, 3], q


def lift_clear(base, pos: np.ndarray, quat: np.ndarray, max_lift_m: float = 0.03) -> tuple[np.ndarray, float]:
    """Raise the cube until it no longer penetrates the (default-pose) sim hand.

    The real cube's pose comes from the cameras, but the sim hand's pose is not the real
    one, so the cube can overlap the sim palm by a few mm; MuJoCo would then shoot it away."""
    m, d = base.model, base.data
    cube_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("task_cube").id}
    adr = base._cube_qpos_adr
    for k in range(int(max_lift_m * 1000) + 1):
        mujoco.mj_resetData(m, d)
        d.qpos[:adr] = base._default_hand_qpos
        d.qpos[adr:adr + 3] = pos + [0, 0, k / 1000]
        d.qpos[adr + 3:adr + 7] = quat
        mujoco.mj_forward(m, d)
        pen = [d.contact[i].dist for i in range(d.ncon)
               if d.contact[i].geom1 in cube_geoms or d.contact[i].geom2 in cube_geoms]
        if not pen or min(pen) > -0.0005:
            return pos + [0, 0, k / 1000], k / 1000
    return pos + [0, 0, max_lift_m], max_lift_m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-from", type=Path, help="snapshot folder (top.png, side.png) with the real cube")
    ap.add_argument("--use-real-position", action="store_true",
                    help="also start the sim cube at the real cube's position (default: orientation only)")
    ap.add_argument("--init", choices=["random", "down"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=2.0, help="sim time of policy motion")
    ap.add_argument("--slow", type=float, default=4.0, help="minimum time stretch")
    ap.add_argument("--max-speed", type=float, default=40.0, help="max joint speed on the real hand, deg/s")
    ap.add_argument("--hz", type=float, default=30.0, help="replay rate (<= 50 for the servo bus)")
    ap.add_argument("--ramp", type=float, default=2.0, help="seconds from neutral into the first pose")
    ap.add_argument("--release", type=float, default=4.0,
                    help="seconds for each of the two release stages at the end: first the thumb goes back "
                         "to neutral (lets go of the cube), then the other fingers. run01 sent every finger "
                         "back in 2 s at once and the cube was flicked off the hand")
    ap.add_argument("--wrist-range", type=float, nargs=2, metavar=("LO", "HI"),
                    help="let the policy move the wrist, kept within [LO, HI] deg in the sim and in the file "
                         "(default: wrist frozen at neutral)")
    ap.add_argument("--out", default="policy_run.yaml", help="file name in the orca replay_sequences folder")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    run_cfg = json.loads((MODEL / "run_config.json").read_text())
    base = OrcaHandRightCubeOrientation(initial_red_face=args.init)
    env = RelativeJointTargetAction(base, max_delta_frac=run_cfg.get("max_delta_frac", 0.05))
    norm = VecNormalize.load(str(MODEL / "vecnormalize.pkl"), DummyVecEnv([lambda: env]))
    norm.training, norm.norm_reward = False, False
    model = PPO.load(str(MODEL / "final_model.zip"), device="cpu")
    bridge = JointBridge(base.model, cfg["joint_ids"])
    wrist_act = bridge.act_of_core["wrist"]

    options = {}
    if args.start_from:
        pos, quat = cube_from_snapshot(args.start_from)
        if args.use_real_position:
            pos2, lift = lift_clear(base, pos, quat)
            options.update(cube_pos=pos2, cube_quat=quat, settle_steps=200)
            print(f"sim cube starts at the real cube: pos {np.round(pos, 4)} (lifted {lift * 1000:.0f} mm), "
                  f"quat {np.round(quat, 3)}")
        else:
            # the training start position (centre of the palm) with the real orientation: placing
            # the cube where the real one sits (near the wrist) lets it roll off the default-pose
            # sim hand, because the real hand's pose there differs
            options.update(cube_quat=quat)
            print(f"sim cube: training start position, real orientation quat {np.round(quat, 3)} "
                  f"(real cube centre was {np.round(pos, 4)})")
    obs, _ = env.reset(seed=args.seed, options=options)
    dt = base.model.opt.timestep * base.frame_skip
    frames, success_at, dropped = [read_core_degrees(bridge, base.data)], None, False
    for k in range(int(args.seconds / dt)):
        action, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
        action = np.asarray(action).reshape(-1).copy()
        if args.wrist_range is None:
            action[wrist_act] = 0.0  # wrist frozen
        else:
            # keep the wrist target (radians; the wrist is not remapped) inside the allowed range
            lo, hi = np.radians(args.wrist_range)
            env._target[wrist_act] = float(np.clip(env._target[wrist_act], lo, hi))
            nxt = env._target[wrist_act] + action[wrist_act] * env.max_delta_frac * env._span[wrist_act]
            if not lo <= nxt <= hi:
                action[wrist_act] = 0.0
        obs, _, terminated, truncated, info = env.step(action)
        frames.append(read_core_degrees(bridge, base.data))
        if success_at is None and info.get("goal_reached"):
            success_at = (k + 1) * dt
        if info.get("dropped"):
            dropped = True
            break
    env.close()
    sim = np.asarray(frames)  # degrees, orca_core joint order, one row per policy step
    print(f"sim: {len(sim)} steps of {dt * 1000:.0f} ms; goal reached "
          f"{'at %.2f s' % success_at if success_at else 'no'}; dropped {dropped}")

    # time stretch: at least --slow, more if a joint would exceed --max-speed
    speed = np.abs(np.diff(sim, axis=0)).max(axis=0) / dt  # deg/s per joint in the sim
    need = float(speed.max() / args.max_speed)
    stretch = max(args.slow, need)
    t_sim = np.arange(len(sim)) * dt * stretch
    t_out = np.arange(0.0, t_sim[-1] + 1e-9, 1.0 / args.hz)
    traj = np.column_stack([np.interp(t_out, t_sim, sim[:, j]) for j in range(sim.shape[1])])

    ids = cfg["joint_ids"]
    neutral = np.array([cfg["neutral_position"].get(j, 0.0) for j in ids], float)
    w = ids.index("wrist")
    wrist_txt = "frozen" if args.wrist_range is None else f"within {args.wrist_range[0]:g}..{args.wrist_range[1]:g} deg"
    if args.wrist_range is None:
        traj[:, w] = neutral[w]
    else:
        traj[:, w] = np.clip(traj[:, w], *args.wrist_range)
    n_ramp = max(1, int(args.ramp * args.hz))
    blend = np.linspace(0, 1, n_ramp)[:, None]
    hold = np.repeat(traj[:1], int(args.hz), axis=0)  # 1 s still at the start pose
    # end: hold 1 s, release the thumb (stage 1), then the other fingers (stage 2)
    n_rel = max(1, int(args.release * args.hz))
    s = np.linspace(0, 1, n_rel)[:, None]
    s = 0.5 - 0.5 * np.cos(np.pi * s)  # ease in / out
    thumb = np.array([j.startswith("thumb") for j in ids])
    mid = traj[-1].copy()
    mid[thumb] = neutral[thumb]
    stage1 = traj[-1] + s * (mid - traj[-1])
    stage2 = mid + s * (neutral - mid)
    full = np.vstack([neutral + blend * (traj[0] - neutral), hold, traj,
                      np.repeat(traj[-1:], int(args.hz), axis=0), stage1, stage2])
    roms = cfg["joint_roms"]
    clipped = 0
    for i, j in enumerate(ids):
        lo, hi = roms[j]
        clipped += int(np.sum((full[:, i] < lo) | (full[:, i] > hi)))
        full[:, i] = np.clip(full[:, i], lo, hi)
    peak = (np.abs(np.diff(full, axis=0)) * args.hz).max(axis=0)
    worst = int(np.argmax(peak))

    out = ORCA / "replay_sequences" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({
        "metadata": {
            "type": "continuous", "created_at": time.strftime("%Y%m%d_%H%M%S"),
            "sampling_frequency_hz": args.hz, "joint_ids": ids, "hand_type": cfg["type"],
            "source": f"orca_sim PPO {MODEL.name} (wrist {wrist_txt}, x{stretch:.1f} slower, "
                      f"start {'real cube' if args.start_from else args.init})",
            "record_mode": "achieved",
        },
        "angles": [[round(float(v), 3) for v in row] for row in full],
    }, sort_keys=False), encoding="utf-8")
    (REPO / "real2sim" / "data" / "policy").mkdir(parents=True, exist_ok=True)
    (REPO / "real2sim" / "data" / "policy" / out.name).write_text(out.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"time stretch x{stretch:.1f} ({'--slow' if stretch == args.slow else 'speed limit'}); "
          f"replay {len(full) / args.hz:.1f} s at {args.hz:g} Hz "
          f"(ramp {args.ramp:g} s + hold 1 s + motion {len(traj) / args.hz:.1f} s + hold 1 s + "
          f"thumb release {args.release:g} s + finger release {args.release:g} s)")
    print(f"fastest joint in the file: {ids[worst]} {peak[worst]:.0f} deg/s (limit {args.max_speed:g}); "
          f"wrist {wrist_txt} (file: {full[:, w].min():.1f}..{full[:, w].max():.1f} deg); "
          f"{clipped} samples clamped into joint_roms")
    print(f"wrote {out}\n\nreplay on the hand (orca folder, Ctrl+C to stop):\n"
          f"  cd C:\\Users\\Owner\\Desktop\\orca\n"
          f"  uv run python scripts/replay_continuous.py orca_core/models/v2/orcahand_right/config.yaml "
          f"--replay-file {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
