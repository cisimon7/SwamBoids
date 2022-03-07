from gymBoidEnv import SwamBoidsEnv
from stable_baselines3 import PPO
import argparse

model_path = "PPO_Training_data/final_model.zip"


def rollout(env, policy, render=False):
    obs = env.reset()

    done = False
    total_reward = 0

    while not done:
        action, _states = policy.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if render:
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
        print("cumulative score #", i, ":", cumulative_score)
        history.append(cumulative_score)
