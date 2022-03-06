from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Set, Dict

import gym
from bindings.BoidModule import Boid
from gym import spaces
from gym.spaces import Space, Box
from ..config import BOID_COUNT
from PyModule.config import WINDOW_WIDTH, WINDOW_HEIGHT

import numpy as np

from ..binaries import Vector2D

ObsvBoid = Box(
    low=np.array([0, 0, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
    high=np.array([WINDOW_WIDTH, WINDOW_HEIGHT, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
    dtype=np.float32
)

ActionBoid = Box(
    low=np.array([-np.inf, -np.inf], dtype=np.float32),
    high=np.array([np.inf, np.inf], dtype=np.float32),
    dtype=np.float32
)  # Action is simply the change in acceleration vector


def split(obs_vec: np.ndarray):
    tup = (obs_vec[:2], obs_vec[2:4], obs_vec[4:])
    return tup


def createObs(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray):
    return dict(
        position=pos,
        velocity=vel,
        acceleration=acc
    )


def np_vector2D(vector: Vector2D) -> np.ndarray:
    return np.array([vector.x, vector.y])


class RenderMode(Enum):
    TRAINING = 1
    EVALUATION = 2


@dataclass
class BoidObject:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def from_boid(self, boid: Boid):
        return BoidObject(
            boid.boid_id,
            np_vector2D(boid.position),
            np_vector2D(boid.velocity),
            np_vector2D(boid.acceleration)
        )
