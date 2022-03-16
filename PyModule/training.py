from datetime import datetime, timedelta
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from gymBoidEnv import SwamBoidsEnv, RenderMode
from config import BOID_COUNT, PREDATOR_COUNT

LOG_DIR = f"trained_models/flocking_algorithm_{BOID_COUNT + PREDATOR_COUNT}_1"
TIME_STEPS = int(2e6)
EVAL_FREQ = 1_000
EVAL_EPISODES = 5


class BoidsEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(BoidsEvalCallback, self).__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            print(f"Current best mean reward: {self.best_mean_reward}")
        return super(BoidsEvalCallback, self)._on_step()


def train():
    logger = configure(folder=LOG_DIR, format_strings=["stdout", "csv", "tensorboard"])

    swam_env = SwamBoidsEnv()
    swam_env.render_mode = RenderMode.TRAINING
    swam_env.evaluation_duration = timedelta(seconds=0, minutes=1)
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
        deterministic=True
    )

    model.set_logger(logger)
    model.learn(total_timesteps=TIME_STEPS, callback=[eval_callback])

    model.save(os.path.join(LOG_DIR, "final_model"))
    env.close()


if __name__ == '__main__':
    start_time = time.perf_counter()
    train()
    end_time = time.perf_counter()

    print(f"Duration:: {end_time - start_time}")
