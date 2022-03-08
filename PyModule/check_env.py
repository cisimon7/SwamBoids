from gym.utils.env_checker import check_env
from gymBoidEnv import SwamBoidsEnv

if __name__ == '__main__':
    env = SwamBoidsEnv()
    env.reset()
    check_env(env)
