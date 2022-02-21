import numpy as np
from typing import NewType, Tuple, List
from dataclasses import dataclass


@dataclass
class Vector2D:
    x: float
    y: float

    def arr(self) -> np.ndarray:
        # Retrieve points in a numpy array format
        return np.array([self.x, self.y])

    @staticmethod
    def from_arr(array: np.ndarray):
        # Create position from array
        assert array.shape[0] == 2
        return Vector2D(array[0], array[1])

    @staticmethod
    def rand(max_x, max_y):
        # Element-wise multiplication
        rand_pos = np.random.rand(2) * np.array([max_x, max_y])
        return Vector2D.from_arr(rand_pos)


Position = Vector2D
Velocity = Vector2D
Acceleration = Vector2D

# Observation Type For Boid. Holds info: Position, Velocity, Acceleration and id of other Boids
ObsBoid = Tuple[Position, Velocity, Acceleration, List[float]]
rnd_obs = (
    Position.from_arr(np.zeros(2)),
    Velocity.from_arr(np.zeros(2)),
    Acceleration.from_arr(np.zeros(2)),
    []
)

# Action is simply the change in acceleration vector
ActionBoid = Vector2D

if __name__ == '__main__':
    pos = Position.from_arr(np.ones(2))
    print(pos)
    print(pos.arr())
