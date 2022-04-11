import multiprocessing
import gym
import neat
import numpy as np

from commons.commons import loadNeatConfig, saveAgent, loadEnvironmentConfigToNeat
import resources.runConfig

runs_per_net = 2
# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []
    env = gym.make(resources.runConfig.envName)
    for _ in range(runs_per_net):
        observation = env.reset()
        fitness = 0.0
        done = False
        while not done:
            action = np.argmax(net.activate(observation))
            observation, reward, done, info = env.step(action)
            fitness += reward
        fitnesses.append(fitness)
    return np.mean(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def createPopulation(config):
    population = neat.Population(config)
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))
    return population

def run():
    loadEnvironmentConfigToNeat()
    config = loadNeatConfig('../resources/neat-config')

    population = createPopulation(config)

    parallelEvaluator = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = population.run(parallelEvaluator.evaluate)

    saveAgent(f"../best-agents/{resources.runConfig.agentName}", winner)

if __name__ == '__main__':
    run()
