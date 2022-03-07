import numpy as np
from ..config import *
from bindings.BoidModule import *
from bindings.FlockModule import *
from bindings.KDTreeModule import *
from bindings.Vector2DModule import *
from bindings.SimulationModule import *


def new_simulation_env(frame_rate: int = FRAME_RATE) -> Simulation:
    return Simulation(
        window_width=WINDOW_WIDTH,
        window_height=WINDOW_HEIGHT,
        boid_size=BOID_SIZE,
        max_speed=MAX_SPEED,
        max_force=MAX_FORCE,
        alignment_weight=ALIGNMENT_WEIGHT,
        cohesion_weight=COHESION_WEIGHT,
        separation_weight=SEPARATION_WEIGHT,
        acceleration_scale=ACCELERATION_SCALE,
        perception=PERCEPTION,
        separation_distance=SEPARATION_DISTANCE,
        noise_scale=NOISE_SCALE,
        fullscreen=False,
        light_scheme=True,
        num_threads=NUM_THREADS,
        frame_rate=frame_rate
    )


def cpp_vec_np(vector: Vector2D) -> np.ndarray:
    """
    Converts the cpp Vector2D class into a numpy array
    :param vector:
    :return:
    """
    return np.array([vector.x, vector.y])
