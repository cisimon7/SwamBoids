from .flocking_algorithm import *


def calculate_reward(m_boid: Boid, neighbors: list[Boid]) -> float:
    """
    Reward function for boid
    :param m_boid: Main boid whose reward function is being calculated
    :param neighbors: List of boids within reachability radius of main boid
    :return: reward score
    """
    # TODO(Doesn't consider predators in implementation)
    cohesion_ = cohesion(m_boid, neighbors)
    separation_ = separation(m_boid, neighbors)
    alignment_ = alignment(m_boid, neighbors)

    # the closer the reward to zero, the better the training
    return (-1 * norm(cohesion_)) + (-1 * norm(separation_)) + (-1 * norm(alignment_))
