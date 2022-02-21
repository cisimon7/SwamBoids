import gym
import numpy as np
from gymBoidEnv import SwamBoidsEnv
from gymBoidEnv.Structs import ActionBoid, ObsBoid

if __name__ == '__main__':

    env = SwamBoidsEnv()
    env.step_render = True

    for i_episode in range(1):
        observation = env.reset()
        for t in range(10):
            env.render()
            print(observation)
            action = ActionBoid.from_arr(np.random.rand(2))
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
