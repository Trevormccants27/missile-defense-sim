import yaml
import pandas as pd
import numpy as np
import argparse
from mds_env import MDSEnv
import matplotlib.pyplot as plt
import warnings

def load_config(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def main(args):
    config = load_config(args.config_file)

    # Create Environment
    env = MDSEnv(config)
    env.reset()

    # Step through environment
    total_reward = 0
    while True:
        action = np.random.random(2)
        print(action)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        # Render plot each step
        if args.render:
            env.render(mode="observation")
            
            # Hold program until plot is closed
            # if done:
            #     plt.show()

        if done:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This file runs a missile defense simulation centered around defending Japanese cities against potential agressors.')

    default_config_file = 'config.yml'

    parser.add_argument('-c', '--config-file', help=f'Specifies location of config file. Default is {default_config_file}', default=default_config_file)
    parser.add_argument('-r', '--render', help=f'Enables rendering plot each time step', action='store_true', default=False)

    args = parser.parse_args()

    warnings.filterwarnings('ignore', '.*Shapely 2.*')

    main(args)