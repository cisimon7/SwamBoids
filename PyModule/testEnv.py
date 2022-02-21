from gymBoidEnv import SwamBoidsEnv, ActionBoid, RenderMode

if __name__ == '__main__':

    env = SwamBoidsEnv()
    env.render_mode = RenderMode.TRAINING
    env.step_render_delay_ms = 300

    for i_episode in range(1):
        env.reset()
        for t in range(50):
            env.render()
            action = ActionBoid(0, 30)  # Constantly move agent up
            observation, reward, done, info = env.step(action)
            # print(observation[0])
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
