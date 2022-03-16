import numpy as np
from ..config import MAX_SPEED, WINDOW_HEIGHT, WINDOW_WIDTH
from .structs import BoidsAction
from ..binaries import Boid, Vector2D, cpp_vec_np


def obs_change(obs: np.ndarray, neighbors: list[Boid], action: BoidsAction) -> np.ndarray:
    """
    This function defines how observation changes given an action:
    :param obs: current observation of boid
    :param neighbors: neighbors of boid in focus
    :param action: predicted action to maximize reward for boid
    :return: new observation
    """
    action /= (1 / 3) * np.linalg.norm(action)
    # return np.r_[obs[:2] + action, obs[2:4] + action, obs[4:6] + action]
    new_vel = limit(obs[2:4] + action, MAX_SPEED)
    result = np.r_[
        pos_constraint(obs[:2] + new_vel),
        new_vel,
        np.zeros(2)
    ]
    return np.asarray(result, dtype=np.float32)


def pos_constraint(position) -> np.ndarray:
    """
    Constraints the given position to be within the toroidal world
    :param position: given position of boid
    :return position: position made to be within the space
    """
    if position[0] < 0:
        position[0] += WINDOW_WIDTH
    if position[1] < 0:
        position[1] += WINDOW_HEIGHT
    if position[0] >= WINDOW_WIDTH:
        position[0] -= WINDOW_WIDTH
    if position[1] >= WINDOW_HEIGHT:
        position[1] -= WINDOW_HEIGHT
    return position


def obs_from_boid(m_boid: Boid) -> np.ndarray:
    """
    Extract the observation vector from boid properties
    :param m_boid: Boid to extra properties from
    :return: numpy array of size 6
    """
    pos = cpp_vec_np(m_boid.position)
    vel = cpp_vec_np(m_boid.velocity)
    acc = cpp_vec_np(m_boid.acceleration)

    # current observation for main boid
    current_obs = np.r_[pos, vel, acc]
    return current_obs


def update_boid(boid: Boid, new_obs: np.ndarray):
    """
    Update boid properties from observation vector
    """
    (n_pos, n_vel, n_acc) = split_to_obs(new_obs)

    c_pos = pos_constraint(n_pos)  # Toroidal world
    boid.position = Vector2D(c_pos[0], c_pos[1])
    boid.velocity = Vector2D(n_vel[0], n_vel[1])
    boid.acceleration = Vector2D(n_acc[0], n_acc[1])

    return boid


def split_to_obs(obs_vec: np.ndarray):
    """
    Splits a vector into components to form observation vector
    :param obs_vec: array of observation vector
    """
    tup = (obs_vec[:2], obs_vec[2:4], obs_vec[4:])
    return tup


def limit(speed: np.ndarray, max_speed) -> np.ndarray:
    x, y = speed
    mag = np.linalg.norm(speed)
    if mag > max_speed:
        x *= max_speed / mag
        y *= max_speed / mag

    return np.array([x, y])


