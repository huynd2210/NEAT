import os
import pickle
import neat
import resources.runConfig
from resources.config import environmentConfigs


def loadNeatConfig(configFilePath):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, configFilePath)
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       config_path)

def loadAgent(filePath):
    with open(filePath, 'rb') as f:
        agent = pickle.load(f)
    print('Loaded agent: ')
    print(agent)
    return agent

def saveAgent(filePath, winner):
    # Save the winner.
    with open(filePath, 'wb') as f:
        pickle.dump(winner, f)
    print('Saved agent: ')
    print(winner)

def replaceLine(line):
    for parameters, values in environmentConfigs[resources.runConfig.envName].items():
        if parameters in line:
            tokens = line.split("=")
            tokens[1] = values
            changedLine = f'{tokens[0]}= {tokens[1]}'
            return changedLine + "\n"
    return line

def loadEnvironmentConfigToNeat():
    neatConfigRead = open("../resources/neat-config", "r")
    configLines = neatConfigRead.readlines()
    newFileContent = "".join(replaceLine(line) for line in configLines)
    with open("../resources/neat-config", "w") as neatConfigWrite:
        neatConfigWrite.write(newFileContent)
    return newFileContent