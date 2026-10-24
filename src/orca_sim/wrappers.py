"""Gymnasium wrappers for training policies on the ORCA environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RelativeJointTargetAction(gym.Wrapper):
    """Delta joint-position control.

    The wrapped env takes absolute joint-position targets in each actuator's
    ``ctrlrange``. This wrapper instead takes an action in ``[-1, 1]^nu`` that
    moves every target by at most ``max_delta_frac`` of its joint range per step,
    starting from the hand's pose at reset. Bounded, normalised actions are what
    PPO/SAC expect, and the rate limit rules out the bang-bang chattering between
    joint limits that absolute-target policies tend to learn.

    The current (normalised) targets are appended to the observation so the
    policy knows where it is commanding the hand to go.
    """

    def __init__(self, env: gym.Env, max_delta_frac: float = 0.05) -> None:
        super().__init__(env)
        base = env.unwrapped
        self._low = np.asarray(base.action_low, dtype=np.float64)
        self._high = np.asarray(base.action_high, dtype=np.float64)
        self._span = np.maximum(self._high - self._low, 1e-6)
        self.max_delta_frac = float(max_delta_frac)
        nu = self._low.shape[0]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(nu,), dtype=np.float32)
        obs_space = env.observation_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space.shape[0] + nu,), dtype=np.float64
        )
        self._target = (self._low + self._high) / 2

    def _normalised_target(self) -> np.ndarray:
        return 2.0 * (self._target - self._low) / self._span - 1.0

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, self._normalised_target()])

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._target = np.clip(self.env.unwrapped.data.ctrl.astype(np.float64), self._low, self._high)
        return self._augment(obs), info

    def step(self, action: np.ndarray):
        delta = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._target = np.clip(
            self._target + delta * self.max_delta_frac * self._span, self._low, self._high
        )
        obs, reward, terminated, truncated, info = self.env.step(self._target.astype(np.float32))
        return self._augment(obs), reward, terminated, truncated, info
