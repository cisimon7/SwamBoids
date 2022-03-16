import gym.spaces
import numpy as np
from enum import Enum
from gym.spaces import Box
from PyModule.binaries import Boid
from dataclasses import dataclass
from ..binaries import cpp_vec_np
from ..config import MAX_FORCE, MAX_SPEED, WINDOW_WIDTH, WINDOW_HEIGHT

# TODO(set low and high to be maximum acceleration per second)
max_acc = MAX_FORCE
max_vel = MAX_SPEED

# A vector of 6 elements: 2 for Position, 2 for Velocity and last2 for Acceleration
# Single Observation space for a boid
# BoidObsSpace = Box(
#     low=np.array(
#         [[0, 0, -max_acc, -max_acc, -max_acc, -max_acc] for _ in range(BOID_COUNT + PREDATOR_COUNT)],
#         dtype=np.float32
#     ),
#     high=np.array(
#         [[WINDOW_WIDTH, WINDOW_HEIGHT, max_acc, max_acc, max_acc, max_acc] for _ in range(BOID_COUNT + PREDATOR_COUNT)],
#         dtype=np.float32
#     ),
#     dtype=np.float32
# )
BoidObsSpace = Box(
    low=np.array(
        [0, 0, -max_vel, -max_vel, -max_acc, -max_acc],
        dtype=np.float32
    ),
    high=np.array(
        [WINDOW_WIDTH, WINDOW_HEIGHT, max_vel, max_vel, max_acc, max_acc],
        dtype=np.float32
    ),
    dtype=np.float32
)

# is an array of size (n x 6) with each row representing the observation vector for a single boid in world
# n = BOID_COUNT + PREDATOR_COUNT
BoidsObservation = np.ndarray

# ActionSpace = Box(
#     low=np.array([[-max_acc, -max_acc] for _ in range(BOID_COUNT + PREDATOR_COUNT)], dtype=np.float32).flatten(),
#     high=np.array([*[[max_acc, max_acc] for _ in range(BOID_COUNT + PREDATOR_COUNT)]], dtype=np.float32).flatten(),
#     dtype=np.float32
# )  # Action is simply the change in acceleration vector

ActionSpace = Box(
    low=np.array([-max_acc, -max_acc], dtype=np.float32),
    high=np.array([max_acc, max_acc], dtype=np.float32),
    dtype=np.float32
)  # Action is simply the change in acceleration vector

# Is an array of size (n x 2) with each row representing action for each boid
BoidsAction = np.ndarray


class RenderMode(Enum):
    TRAINING = 1
    EVALUATION = 2


@dataclass
class BoidObject:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    @staticmethod
    def from_boid(boid: Boid):
        return BoidObject(
            boid.boid_id,
            cpp_vec_np(boid.position),
            cpp_vec_np(boid.velocity),
            cpp_vec_np(boid.acceleration)
        )
