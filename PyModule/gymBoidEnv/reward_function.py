from .flocking_algorithm import *


def calculate_reward(m_boid: Boid, neighbors: list[Boid]) -> float:
    """
    Reward function for boid
    :param m_boid: Main boid whose reward function is being calculated
    :param neighbors: List of boids within reachability radius of main boid
    :return: reward score
    """
    # TODO(Doesn't consider predators in implementation)
    cohesion_vector = cohesion(m_boid, neighbors)
    separation_vector = separation(m_boid, neighbors)
    alignment_vector = alignment(m_boid, neighbors)

    # the closer the reward to zero, the better the training
    return -(cohesion_vector + separation_vector + alignment_vector)
