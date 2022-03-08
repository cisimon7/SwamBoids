import argparse
from stable_baselines3 import PPO
from gymBoidEnv import SwamBoidsEnv, RenderMode

model_path = "trained_models/flocking_algorithm1/final_model.zip"


def rollout(env: SwamBoidsEnv, policy, render=False):
    env.render_mode = RenderMode.EVALUATION
    env.step_render_delay_ms = 5  # Delay between simulation
    obs = env.reset()
    total_reward = 0

    done = False

    # run for a period of time
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
    parser.add_argument('--render', action='store_true', help='render to screen?', default=True)
    args = parser.parse_args()

    boid_env = SwamBoidsEnv()

    render_mode = args.render
    ppo_model = PPO.load(args.model_path, env=boid_env)

    history = []
    for i in range(1):
        cumulative_score = rollout(boid_env, ppo_model, render_mode)
        print(f"Run #: {i} -> {cumulative_score}")
        history.append(cumulative_score)
