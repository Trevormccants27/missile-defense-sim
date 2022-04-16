import yaml
import pandas as pd
import numpy as np
import argparse
from mds_env import MDSEnv
import matplotlib.pyplot as plt
import warnings
from actors import *
from glob import glob
import os

def load_config(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def main(args):
    config_file_list = glob(args.config_file)

    if args.save_images and os.path.isdir(args.save_location):
        raise Exception(f'Save location already exists: \"{args.save_location}\"! Please specify a new save location.')

    final_results = pd.DataFrame(columns=['config_file', 'total_reward'])
    for config_num, config_file in enumerate(config_file_list):
        print(f'------------- RUNNING CONFIG FILE: {config_file} -------------')
        save_location = os.path.join(args.save_location, os.path.basename(config_file.split('.')[0]))
        
        if args.save_images:
            os.makedirs(save_location)
            
        config = load_config(config_file)

        # Create Environment
        env = MDSEnv(config)
        env.reset(config['seed'])

        # Initialize actor
        actors = {}
        for nation in ['US', 'Japan']:
            if config[nation]['actor_type'] == 'random':
                actors[nation] = actor(env.action_space, env.threat_platforms, env.defended_assets, config['placement_lat_band'], config['placement_lng_band'])
            elif config[nation]['actor_type'] == 'in_between':
                actors[nation] = actor_in_between(env.action_space, env.threat_platforms, env.defended_assets, config['placement_lat_band'], config['placement_lng_band'])
            elif config[nation]['actor_type'] == 'near_defense':
                actors[nation] = actor_near_defense(env.action_space, env.threat_platforms, env.defended_assets, config['placement_lat_band'], config['placement_lng_band'])
            elif config[nation]['actor_type'] == 'spaced_out':
                actors[nation] = actor_spaced_out(env.action_space, env.threat_platforms, env.defended_assets, config['placement_lat_band'], config['placement_lng_band'])
            elif config[nation]['actor_type'] == 'in_between_and_near_defense':
                actors[nation] = actor_in_between_and_near_defense(env.action_space, env.threat_platforms, env.defended_assets, config['placement_lat_band'], config['placement_lng_band'])
            else:
                raise Exception(f"Invalid actor type selected: {config['actor_type']}")

        # Step through environment
        total_reward = 0
        for t in range(config['max_time_steps']):
            print(f'Current time step: {t}')

            for nation, actor_obj in actors.items():
                action = actor_obj.get_action()
                obs, reward, done, _ = env.step(action, nation)
                total_reward += reward

            # Render plot each step
            if args.render or args.save_images:
                env.render(mode=args.render_mode)
                
                if args.save_images:
                    plt.savefig(os.path.join(save_location, f'{t}.png'))

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
    default_save_location = './plots'

    parser.add_argument('-c', '--config-file', help=f'Specifies location of config file. Note this can also contain wild cards to specify a list. Default is {default_config_file}', default=default_config_file)
    parser.add_argument('-r', '--render', help=f'Enables rendering plot each time step', action='store_true', default=False)
    parser.add_argument('-s', '--save-images', help=f'Enables saving images each time step', action='store_true', default=False)
    parser.add_argument('--save-location', help=f'Specifies location to save plots. Default is {default_save_location}', default=default_save_location)
    parser.add_argument('-m', '--render-mode', help=f'Specifies which mode to render in. Options are \"human\" or \"observation\". Default is {default_render_mode}', default=default_render_mode)

    args = parser.parse_args()

    warnings.filterwarnings('ignore', '.*Shapely 2.*')

    main(args)