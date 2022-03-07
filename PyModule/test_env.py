import numpy as np
from gymBoidEnv import SwamBoidsEnv, RenderMode
from config import BOID_COUNT, PREDATOR_COUNT, WINDOW_WIDTH, WINDOW_HEIGHT

if __name__ == '__main__':

    env = SwamBoidsEnv()
    env.render_mode = RenderMode.EVALUATION
    env.step_render_delay_ms = 5  # Delay between simulation

    for i_episode in range(1):
        env.reset()
        for t in range(1_000):
            env.render()

            # Constantly move all boids down
            actions = np.asarray([*[np.random.randn(2) for _ in range(BOID_COUNT + PREDATOR_COUNT)]])
            observation, reward, done, info = env.step(actions)
            if done:
                print(f"Episode finished after {t + 1} time steps")
                break
    env.close()
