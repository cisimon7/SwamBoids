import datetime
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.vec_env import SubprocVecEnv

from gymBoidEnv import SwamBoidsEnv, RenderMode

LOG_DIR = "trained_models/flocking_algorithm"
TIME_STEPS = int(2e6)
EVAL_FREQ = 1_000
EVAL_EPISODES = 5


class BoidsEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(BoidsEvalCallback, self).__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        return super(BoidsEvalCallback, self)._on_step()


def train():
    logger = configure(folder=LOG_DIR, format_strings=["stdout", "csv", "tensorboard"])

    swam_env = SwamBoidsEnv()
    swam_env.render_mode = RenderMode.TRAINING
    swam_env.evaluation_duration = datetime.timedelta(seconds=0, minutes=1)
    swam_env.step_render_delay_ms = 5  # Delay between simulation

    env = Monitor(swam_env, LOG_DIR)

    model = PPO(
        "MlpPolicy",
        env=env,
        learning_rate=(lambda rate_left: rate_left * 1e-4),
        n_steps=TIME_STEPS,
        batch_size=1024,
        verbose=2,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.0
    )

    eval_callback = BoidsEvalCallback(
        eval_env=env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=False
    )

    model.set_logger(logger)
    model.learn(total_timesteps=TIME_STEPS, callback=[eval_callback])

    model.save(os.path.join(LOG_DIR, "final_model"))
    env.close()


if __name__ == '__main__':
    start_time = time.perf_counter()
    train()
    end_time = time.perf_counter()

    print(f"Duration: {end_time - start_time}")
