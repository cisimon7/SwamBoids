from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List

import numpy as np

from ..binaries import Vector2D


@dataclass
class Vec2D:
    x: float
    y: float

    def arr(self) -> np.ndarray:
        # Retrieve points in a numpy array format
        return np.array([self.x, self.y])

    def __add__(self, other):
        return Vec2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Vec2D(self.x * other.x, self.y * other.y)

    @staticmethod
    def from_arr(array: np.ndarray):
        # Create position from array
        assert array.shape[0] == 2
        return Vec2D(array[0], array[1])

    @staticmethod
    def rand(max_x, max_y):
        # Element-wise multiplication
        rand_pos = np.random.rand(2) * np.array([max_x, max_y])
        return Vec2D.from_arr(rand_pos)

    @staticmethod
    def from_vector2D(vec: Vector2D):
        return Vec2D(vec.x, vec.y)


Pos = Vec2D  # Position vector
Vel = Vec2D  # Velocity vector
Acc = Vec2D  # Acceleration vector

# Observation Type For Boid. Holds info: Position, Velocity, Acceleration and id of other Boids
ObsBoid = Tuple[Pos, Vel, Acc, List[float]]
rnd_obs = (
    Pos.from_arr(np.zeros(2)),
    Vel.from_arr(np.zeros(2)),
    Acc.from_arr(np.zeros(2)),
    []
)

# Action is simply the change in acceleration vector
ActionBoid = Vec2D


# Render Mode
class RenderMode(Enum):
    TRAINING = 1
    EVALUATION = 2
