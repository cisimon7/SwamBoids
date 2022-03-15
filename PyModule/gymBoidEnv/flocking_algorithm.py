import numpy as np
from PyModule.binaries import Boid, cpp_vec_np
from PyModule.config import WINDOW_WIDTH, WINDOW_HEIGHT, PERCEPTION


# PERCEPTION is the maximum distance between a boid and any of its neighbor boid

def cohesion(m_boid: Boid, neighbors: list[Boid]) -> float:
    if len(neighbors) == 0:
        return norm(np.zeros(2))
    else:
        positions = list(map(lambda boid_: cpp_vec_np(boid_.position), neighbors))
        center = np.sum(np.asarray(positions), axis=0) / len(positions)

        boid_pos = cpp_vec_np(m_boid.position)
        dis_to_center = toroidal_difference(boid_pos, center) / PERCEPTION
        return norm(dis_to_center)


def separation(m_boid: Boid, neighbors: list[Boid]) -> float:
    if len(neighbors) == 0:
        return norm(np.zeros(2))
    else:
        distances = list(map(
            lambda boid_: toroidal_difference(cpp_vec_np(boid_.position), cpp_vec_np(m_boid.position)),
            neighbors
        ))

        # Error
        error_distances = list(map(
            lambda dist: (dist - PERCEPTION * normalize(dist)) / PERCEPTION,
            distances
        ))

        error = list(map(
            lambda err: norm(err),
            error_distances
        ))

        return np.sum(error)


def alignment(m_boid: Boid, neighbors: list[Boid]) -> float:
    if len(neighbors) == 0:
        return norm(np.zeros(2))
    else:
        velocities = list(map(lambda boid_: cpp_vec_np(boid_.velocity), neighbors))

        # Normalize velocity to get direction
        directions = np.asarray([normalize(velocity) for velocity in velocities])
        avg_direction = normalize(np.sum(directions, axis=0))

        boid_direction = normalize(cpp_vec_np(m_boid.velocity))

        align_diff = boid_direction - avg_direction

        return norm(align_diff / 2)


def toroidal_difference(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    assert (vec1.shape == vec2.shape == (2,)), "Unequal vector length"
    (dx, dy) = vec1 - vec2

    if dx > WINDOW_WIDTH / 2:
        dx = WINDOW_WIDTH - dx

    if dy > WINDOW_HEIGHT / 2:
        dy = WINDOW_HEIGHT - dy

    return np.array([dx, dy])


def normalize(vec: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(vec)
    if mag == 0:
        return vec
    return vec / mag


def norm(vec: np.ndarray):
    return np.linalg.norm(vec)

# if __name__ == '__main__':
#     error = square_error(np.array([1, 1, 1]), np.array([0, 0, 0]))
#     print(error)
#
#     print(np.sum(np.array([[1, 1], [2, 2], [3, 3]]), axis=0))
#     print(np.linalg.norm(np.ones(2)))
#     print(np.ones(2).shape)
#     print(np.ones(2) / norm(np.zeros(2)))
#
#     vectors = np.ones((3, 2))
#     print(np.sum(vectors, axis=0) / len(vectors))

# Explanation of flocking algorithm:
# https://medium.com/swlh/boids-a-simple-way-to-simulate-how-birds-flock-in-processing-69057930c229
