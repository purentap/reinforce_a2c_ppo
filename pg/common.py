""" Loggers and Atari Wrappers
    You don't need to modify this section.
"""
from typing import Dict, Tuple, Any
import gym
import numpy as np
import os
import datetime
import csv
import tempfile


class PrintWriter():
    """ Simple console writer """

    def __init__(self, end_line: str = "\n", flush: bool = False):
        self.end_line = end_line
        self.flush = flush

    def __call__(self, field_dict: Dict[str, float]) -> None:
        print(
            ", ".join("{:10}: {:6}".format(key, value) for key, value in field_dict.items()),
            end=self.end_line,
            flush=self.flush,
        )


class CSVwriter():
    """ CSV writer for logging and plotting """

    def __init__(self, log_dir: str):
        if log_dir is None:
            log_dir = tempfile.TemporaryDirectory().name
        self.field_keys = None
        log_dir = self.log_path = os.path.join(
            log_dir,
            datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S"),
        )
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "progress.csv")
        print("Logging at: {}".format(log_dir))

    def __call__(self, field_dict: Dict[str, float]) -> None:
        # Lazy initialization
        if self.field_keys is None:
            self.field_keys = list(field_dict.keys())
            with open(self.log_path, "a") as fobj:
                writer = csv.DictWriter(fobj, fieldnames=self.field_keys)
                writer.writeheader()
        with open(self.log_path, "a") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=self.field_keys)
            writer.writerow(field_dict)


class ResizeAndScalePong(gym.ObservationWrapper):
    """ Observation wrapper that is designed for Pong by Andrej Karphaty.
    Crop, rescale, transpose and simplify the state.
    """

    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, 80, 80), dtype=np.int8)

    # From Andrew Karphaty
    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs[35:195]  # crop
        obs = obs[::2, ::2, 0:1]  # downsample by factor of 2
        obs.transpose(0, 1, 2)
        obs = obs.transpose(2, 0, 1)
        obs[obs == 144] = 0  # erase background (background type 1)
        obs[obs == 109] = 0  # erase background (background type 2)
        obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1
        obs = obs.astype(np.int8)
        return obs


class DerivativeEnv(gym.Wrapper):
    """ Environment wrapper that returns the difference between two consecutive
    observations """

    def reset(self, **kwargs):
        self.pre_obs = self.env.reset(**kwargs)
        return np.zeros(self.observation_space.shape,
                        dtype=self.observation_space.dtype)

    def step(self, ac: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        cur_obs, *remaining = self.env.step(ac)
        obs = cur_obs - self.pre_obs
        self.pre_obs = cur_obs
        return (obs, *remaining)


# From Open AI Baseline
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, noop_max: int = 30, override_num_noops: bool = None):
        """ Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = override_num_noops
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs) -> np.ndarray:
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        return self.env.step(ac)


class DoubleActionPong(gym.Wrapper):
    """ Pong specific environment wrapper that reduces the action space into
    2 with only meaningful actions (up and down) """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.action_space = gym.spaces.Discrete(2)

    def step(self, ac: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        return self.env.step(ac + 2)

    def reset(self, **kwargs) -> np.ndarray:
        return self.env.reset(**kwargs)
