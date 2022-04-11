import gym

import resources.runConfig

def test_environment(env, do_print=False):
    observation = env.reset()
    if do_print:
        print("Observation Size: ", len(observation))
        print("Observation: ", observation)
        print("Possible Action space: ", env.action_space)
        print(type(env.action_space))

    done = False
    while not done:
        observation, reward, done, info = env.step(env.action_space.sample())
        env.render()


if __name__ == '__main__':
    # env = gym.make("CartPole-v1")
    env = gym.make(resources.runConfig.envName)

    test_environment(env, True)
