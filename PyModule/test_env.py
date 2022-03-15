from gymBoidEnv import SwamBoidsEnv, RenderMode

if __name__ == '__main__':
    # Test Environment

    env = SwamBoidsEnv()
    env.render_mode = RenderMode.EVALUATION
    env.step_render_delay_ms = 100  # Delay between simulation

    for i_episode in range(1):
        env.reset()
        for t in range(1_000):
            env.render()

            actions = env.action_space.sample()
            observation, reward, done, info = env.step(actions)
            if done:
                print(f"Episode finished after {t + 1} time steps")
                break
    env.close()
