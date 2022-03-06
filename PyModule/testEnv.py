import pickle
import numpy as np
from bindings.BoidModule import Boid

from PyModule.gymBoidEnv import SwamBoidsEnv
from stable_baselines3.common.env_checker import check_env
from PyModule.gymBoidEnv.Structs import RenderMode

# if __name__ == '__main__':
#     env = SwamBoidsEnv()
#     check_env(env)

if __name__ == '__main__':

    env = SwamBoidsEnv()
    env.render_mode = RenderMode.TRAINING
    env.step_render_delay_ms = 5  # Delay between simulation

    for i_episode in range(1):
        env.reset()
        for t in range(2_000):
            env.render()
            action = np.array([0, 1])  # Constantly move agent up
            observation, reward, done, info = env.step(action)
            # print(observation[0])
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
