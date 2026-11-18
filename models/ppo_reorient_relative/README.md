# PPO cube reorientation policy (relative joint control)

Final model of the `ppo_reorient_random_relative` run: 10M steps, random start
orientation, `RelativeJointTargetAction` (max 5% of joint range per step).

Evaluation under the held success rule (red face up within 15 deg, held 25 steps):
80% from random starts (200 episodes), 100% from red-face-down, 0% drops,
median 0.71 s to success.

    python scripts/watch_policy.py models/ppo_reorient_relative --checkpoint final
    python scripts/plot_training.py models/ppo_reorient_relative --once --save curves.png

`vecnormalize.pkl` holds the observation normalisation stats; the policy needs it.
