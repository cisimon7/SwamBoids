from gym import Env
import numpy as np
from typing import Union, Tuple, TypeVar

from EnvStructs import BoidAction, BoidObs


class BoidEnv(Env):
    def __init__(self):
        pass

    def step(self, action: BoidAction) -> Tuple[BoidObs, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, logging, and sometimes learning)
        """
        return BoidObs(), 0.0, False, {}

    def reset(self):
        pass

    def render(self, mode="human"):
        pass
