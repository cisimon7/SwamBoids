import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from gymBoidEnv import SwamBoidsEnv, RenderMode

LOG_DIR = "trained_models/flocking_algorithm"
TIME_STEPS = 200  # int(2e8)
EVAL_FREQ = 1_000
EVAL_EPISODES = 5


def train():
    logger = configure(folder=LOG_DIR, format_strings=["stdout", "csv", "tensorboard"])

    swam_env = SwamBoidsEnv()
    swam_env.render_mode = RenderMode.TRAINING
    swam_env.step_render_delay_ms = 5  # Delay between simulation

    env = Monitor(swam_env, LOG_DIR)

    model = PPO("MlpPolicy", env, learning_rate=(lambda rate_left: rate_left * 1e-4), n_steps=TIME_STEPS,
                batch_size=1024, verbose=2, gae_lambda=0.95, gamma=0.99, ent_coef=0.0)

    eval_callback = EvalCallback(env, best_model_save_path=LOG_DIR, log_path=LOG_DIR, eval_freq=EVAL_FREQ,
                                 n_eval_episodes=EVAL_EPISODES)
    model.set_logger(logger)
    model.learn(total_timesteps=TIME_STEPS, callback=[eval_callback])

    model.save(os.path.join(LOG_DIR, "final_model"))  # probably never get to this point.
    env.close()


if __name__ == '__main__':
    start_time = time.perf_counter()
    train()
    end_time = time.perf_counter()

    print(f"Duration: {end_time - start_time}")
