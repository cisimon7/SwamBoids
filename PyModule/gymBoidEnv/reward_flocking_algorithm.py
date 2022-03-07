import numpy as np
from PyModule.binaries import Boid, cpp_vec_np
from PyModule.config import WINDOW_WIDTH, WINDOW_HEIGHT


# Doesn't consider predators in implementation

def calculate_reward(m_boid: Boid, neighbors: list[Boid]) -> float:
    """
    Reward function for boid
    :param m_boid: Main boid whose reward function is being calculated
    :param neighbors: List of boids within reachability radius of main boid
    :return: reward score
    """
    cohesion_ = cohesion(m_boid, neighbors)
    separation_ = separation(m_boid, neighbors)
    alignment_ = alignment(m_boid, neighbors)

    # the closer the reward to zero, the better the training
    return (-1 * norm(cohesion_)) + (-1 * norm(separation_)) + (-1 * norm(alignment_))


def cohesion(m_boid: Boid, neighbors: list[Boid]):
    """
    Calculates the root-squared difference between the center of the neighbor boids and the position of the m_boid
    :param m_boid:
    :param neighbors:
    :return: error
    """
    if len(neighbors) == 0:
        return np.zeros(2)
    else:
        center = cal_center(neighbors)
        boid_pos = cpp_vec_np(m_boid.position)
        dis_to_center = boid_pos - center
        return dis_to_center


def separation(m_boid: Boid, neighbors: list[Boid]):
    if len(neighbors) == 0:
        return np.zeros(2)
    else:
        distances = list(map(
            lambda boid_: toroidal_distance(cpp_vec_np(boid_.position), cpp_vec_np(m_boid.position)),
            neighbors
        ))

        sum_dist = np.sum(np.asarray(distances), axis=0)
        return sum_dist


def alignment(m_boid: Boid, neighbors: list[Boid]):
    if len(neighbors) == 0:
        return np.zeros(2)
    else:
        velocities = list(map(lambda boid_: cpp_vec_np(boid_.velocity), neighbors))
        neighbors_avg_direction = np.sum(velocities, axis=0)
        avg_direction = neighbors_avg_direction
        boid_direction = (cpp_vec_np(m_boid.velocity))

        align_diff = boid_direction - avg_direction

        return align_diff


def square_error(vec1: np.ndarray, vec2: np.ndarray):
    """
    Root-Squared error between two vectors
    :param vec1:
    :param vec2:
    :return:
    """
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def cal_center(neighbors: list[Boid]):
    """
    Finds the center of the boids simply by taking the mean of the positions
    :param neighbors: list of boids
    :return: mean of positions
    """
    positions = list(map(lambda boid_: cpp_vec_np(boid_.position), neighbors))
    return np.sum(np.asarray(positions), axis=0) / len(positions)


def toroidal_distance(vec1: np.ndarray, vec2: np.ndarray):
    assert (vec1.shape == vec2.shape == (2,)), "Unequal vector length"
    (dx, dy) = vec1 - vec2

    if dx > WINDOW_WIDTH / 2:
        dx = WINDOW_WIDTH - dx

    if dy > WINDOW_HEIGHT / 2:
        dy = WINDOW_HEIGHT - dy

    return np.array([dx, dy])


def normalize(vec: np.ndarray):
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

# Explanation of flocking algorithm:
# https://medium.com/swlh/boids-a-simple-way-to-simulate-how-birds-flock-in-processing-69057930c229
