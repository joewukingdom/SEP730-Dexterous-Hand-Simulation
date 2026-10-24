"""Train a PPO policy on the ORCA cube-reorientation task (red face up).

    python scripts/train_ppo_reorient.py                       # defaults
    python scripts/train_ppo_reorient.py --steps 20_000_000 --n-envs 16
    python scripts/train_ppo_reorient.py --init down           # harder: always start red-face-down
    python scripts/train_ppo_reorient.py --resume runs/<old>/checkpoints/ppo_5500000_steps.zip
                                         # continue from a checkpoint (its normalisation stats are reused)

Outputs go to runs/ppo_reorient_<timestamp>/:
    progress.csv       training curves (reward, losses, eval success rate)
    best_model.zip     policy with the best mean eval reward so far
    final_model.zip    policy at the end of training
    vecnormalize.pkl   observation/reward normalisation stats (needed to run the policy)
    checkpoints/       model + normalisation stats every --checkpoint-every steps, so a
                       run in progress can be watched with scripts/watch_policy.py

Success is the *held* criterion from OrcaHandRightCubeOrientation: red face within
15 deg of up while the cube is nearly still, for 25 consecutive steps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)


def make_env(init: str, seed: int, action: str = "relative", max_delta_frac: float = 0.05):
    def _thunk():
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from orca_sim import OrcaHandRightCubeOrientation
        from orca_sim.wrappers import RelativeJointTargetAction

        env = OrcaHandRightCubeOrientation(initial_red_face=init)
        if action == "relative":
            env = RelativeJointTargetAction(env, max_delta_frac=max_delta_frac)
        env = Monitor(env, info_keywords=("is_success",))
        env.reset(seed=seed)
        return env

    return _thunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=10_000_000, help="total environment steps")
    parser.add_argument("--n-envs", type=int, default=16, help="parallel simulation processes")
    parser.add_argument("--init", choices=["random", "down"], default="random",
                        help="initial cube orientation: random non-solved, or always red-face-down")
    parser.add_argument("--eval-every", type=int, default=200_000, help="env steps between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=500_000,
                        help="env steps between saved checkpoints (model + normalisation stats)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="checkpoint .zip to continue from; --steps are then added on top of it")
    parser.add_argument("--action", choices=["relative", "absolute"], default="relative",
                        help="relative: [-1,1] deltas on joint targets (rate-limited, smoother); "
                             "absolute: raw joint-position targets")
    parser.add_argument("--max-delta-frac", type=float, default=0.05,
                        help="relative mode: max target change per step, as a fraction of joint range")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or REPO_ROOT / "runs" / f"ppo_reorient_{args.init}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(
        {"init": args.init, "action": args.action, "max_delta_frac": args.max_delta_frac}, indent=2))
    env_kwargs = dict(action=args.action, max_delta_frac=args.max_delta_frac)

    train_env = SubprocVecEnv([make_env(args.init, args.seed + i, **env_kwargs) for i in range(args.n_envs)])
    if args.resume:
        steps = args.resume.stem.split("_")[1]
        norm_path = args.resume.with_name(f"ppo_vecnormalize_{steps}_steps.pkl")
        train_env = VecNormalize.load(str(norm_path), train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    # Eval env shares the training normalisation stats but does not update them.
    eval_env = DummyVecEnv([make_env(args.init, 10_000 + args.seed, **env_kwargs)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    class SyncNormEval(EvalCallback):
        def _on_step(self) -> bool:
            self.eval_env.obs_rms = self.training_env.obs_rms
            return super()._on_step()

    eval_cb = SyncNormEval(
        eval_env,
        n_eval_episodes=args.eval_episodes,
        eval_freq=max(1, args.eval_every // args.n_envs),
        best_model_save_path=str(out),
        log_path=str(out),
        deterministic=True,
    )

    ppo_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        n_steps=512,             # per env -> 8192 samples per update with 16 envs
        batch_size=2048,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                           log_std_init=-0.5 if args.action == "relative" else 0.0),
        device="cpu",            # tiny MLP: CPU is as fast as GPU and avoids transfer overhead
        seed=args.seed,
        verbose=0,
    )
    if args.resume:
        for key in ("policy", "policy_kwargs", "env"):
            ppo_kwargs.pop(key)
        model = PPO.load(str(args.resume), env=train_env, **ppo_kwargs)
        print(f"Resuming from {args.resume} ({model.num_timesteps:,} steps)")
    else:
        model = PPO(**ppo_kwargs)
    model.set_logger(configure(str(out), ["stdout", "csv"]))

    print(f"Training PPO for {args.steps:,} steps on {args.n_envs} envs (init={args.init}, action={args.action}) -> {out}")
    t0 = time.time()
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // args.n_envs),
        save_path=str(out / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )
    model.learn(total_timesteps=args.steps, callback=[eval_cb, ckpt_cb],
                reset_num_timesteps=not args.resume)
    model.save(out / "final_model")
    train_env.save(str(out / "vecnormalize.pkl"))
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Eval success rate over time is the "
          f"eval/success_rate column of {out / 'progress.csv'}.")


if __name__ == "__main__":
    main()
