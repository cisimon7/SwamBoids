import argparse
from datetime import datetime, timedelta

from stable_baselines3 import PPO

from gymBoidEnv import SwamBoidsEnv

model_path = "trained_models/flocking_algorithm/final_model.zip"


def rollout(env: SwamBoidsEnv, policy, render=False):
    obs = env.reset()
    total_reward = 0

    stop = datetime.now() + timedelta(seconds=5, minutes=0)

    # run for a period of time
    while datetime.now() < stop:
        action, _states = policy.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        total_reward += reward

        if render_mode:
            env.render()

    return total_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate pre-trained PPO agent.')
    parser.add_argument('--model-path', help='path to stable-baselines model.', type=str, default=model_path)
    parser.add_argument('--render', action='store_true', help='render to screen?', default=False)
    args = parser.parse_args()

    boid_env = SwamBoidsEnv()

    render_mode = args.render
    model = PPO.load(args.model_path, env=boid_env)

    history = []
    for i in range(1):
        cumulative_score = rollout(boid_env, model, render_mode)
        print(f"Run #: {i} -> {cumulative_score}")
        history.append(cumulative_score)
