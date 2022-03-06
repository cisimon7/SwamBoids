import os
import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from gymBoidEnv import SwamBoidsEnv, ActionBoid, RenderMode

"""
QUESTIONS:
    ✹ What should be in the observation vector?
    ✹ What is the action vector?
    ✹ How does action vector give new observation?
    ✹ How many boids and predators should be in training environment?
    ✹ How should cohesion, alignment and separation affect reward?
    ✹ How should the other boids behave in order not to make the task of training the boid too difficult?
    ✹ ?
"""
LOG_DIR = "PPO_Training_data"
TIME_STEPS = 1


def train():
    logger = configure(folder=LOG_DIR, format_strings=["stdout", "csv", "tensorboard"])

    swam_env = SwamBoidsEnv()
    swam_env.render_mode = RenderMode.TRAINING
    swam_env.step_render_delay_ms = 5  # Delay between simulation

    env = Monitor(swam_env, LOG_DIR)

    ppo_model = PPO("MlpPolicy", env, learning_rate=(lambda rate_left: rate_left * 1e-4), n_steps=1024,
                    batch_size=1024, verbose=2, gae_lambda=0.95, gamma=0.99, ent_coef=0.0)

    ppo_model.learn(total_timesteps=TIME_STEPS, callback=[])
    ppo_model.save(os.path.join(LOG_DIR, "final_model"))  # probably never get to this point.
    env.close()


if __name__ == '__main__':
    start_time = time.perf_counter()
    train()
    end_time = time.perf_counter()

    print(f"Duration: {end_time - start_time}")
