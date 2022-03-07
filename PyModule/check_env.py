import numpy as np
from gym.utils.env_checker import check_env

from PyModule.gymBoidEnv import ActionSpace
from gymBoidEnv import SwamBoidsEnv

if __name__ == '__main__':
    env = SwamBoidsEnv()
    env.reset()

    check_env(env)

    # a_space = ActionSpace
    # print(a_space.shape)
    # print(a_space.sample().shape)
    # print(np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 2))
