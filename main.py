import yaml
import pandas as pd
import numpy as np
import argparse
from mds_env import MDSEnv
import matplotlib.pyplot as plt
import warnings
from actors import *
from glob import glob

def load_config(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def main(args):
    config_file_list = glob(args.config_file)

    final_results = pd.DataFrame(columns=['config_file', 'total_reward'])
    for config_num, config_file in enumerate(config_file_list):
        print(f'------------- RUNNING CONFIG FILE: {config_file} -------------')
        config = load_config(config_file)

        # Create Environment
        env = MDSEnv(config)
        env.reset(config['seed'])

        # Initialize actor
        if config['actor_type'] == 'random':
            actor_obj = actor(env.action_space, env.threat_platforms, env.defended_assets, config['placement_lat_band'], config['placement_lng_band'])
        elif config['actor_type'] == 'in_between':
            actor_obj = actor_in_between(env.action_space, env.threat_platforms, env.defended_assets, config['placement_lat_band'], config['placement_lng_band'])
        elif config['actor_type'] == 'near_defense':
            actor_obj = actor_near_defense(env.action_space, env.threat_platforms, env.defended_assets, config['placement_lat_band'], config['placement_lng_band'])
        elif config['actor_type'] == 'spaced_out':
            actor_obj = actor_spaced_out(env.action_space, env.threat_platforms, env.defended_assets, config['placement_lat_band'], config['placement_lng_band'])
        else:
            raise Exception(f"Invalid actor type selected: {config['actor_type']}")

        # Step through environment
        total_reward = 0
        for t in range(config['max_time_steps']):
            print(f'Current time step: {t}')

            action = actor_obj.get_action()
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            # Render plot each step
            if args.render:
                env.render(mode=args.render_mode)

            if done:
                break

        print(f'Total reward: {total_reward}')
        final_results.loc[config_num] = [config_file, total_reward]

    print(final_results)
    final_results.to_csv('final_results.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This file runs a missile defense simulation centered around defending Japanese cities against potential agressors.')

    default_config_file = 'config*.yml'
    default_render_mode = 'human'

    parser.add_argument('-c', '--config-file', help=f'Specifies location of config file. Note this can also contain wild cards to specify a list. Default is {default_config_file}', default=default_config_file)
    parser.add_argument('-r', '--render', help=f'Enables rendering plot each time step', action='store_true', default=False)
    parser.add_argument('-m', '--render-mode', help=f'Specifies which mode to render in. Options are \"human\" or \"observation\". Default is {default_render_mode}', default=default_render_mode)

    args = parser.parse_args()

    warnings.filterwarnings('ignore', '.*Shapely 2.*')

    main(args)