from gymBoidEnv import SwamBoidsEnv, ActionBoid, RenderMode

if __name__ == '__main__':

    env = SwamBoidsEnv()
    env.render_mode = RenderMode.TRAINING
    env.step_render_delay_ms = 5  # Delay between simulation

    for i_episode in range(1):
        env.reset()
        for t in range(1_000):
            env.render()
            action = ActionBoid(0, 1)  # Constantly move agent up
            observation, reward, done, info = env.step(action)
            # print(observation[0])
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
