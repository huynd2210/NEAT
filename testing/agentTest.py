import gym
import neat
import numpy as np

import resources.runConfig
from commons.commons import loadNeatConfig, loadAgent, loadEnvironmentConfigToNeat
from resources.config import environmentConfigs


def getAction(net, observation):
    return np.argmax(net.activate(observation)) \
        if eval(environmentConfigs[resources.runConfig.envName]["is_argmax"]) \
        else net.activate(observation)


def runAgent(agent, config, env):
    net = neat.nn.FeedForwardNetwork.create(agent, config)
    observation = env.reset()
    done = False
    while not done:
        action = getAction(net, observation)
        observation, reward, done, info = env.step(action)
        env.render()


def testAgent():
    loadEnvironmentConfigToNeat()
    env = gym.make(resources.runConfig.envName)
    config = loadNeatConfig('../resources/neat-config')
    agent = loadAgent(f'../best-agents/{resources.runConfig.agentName}')
    runAgent(agent, config, env)
