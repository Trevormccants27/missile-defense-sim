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
from tqdm import tqdm
from tqdm.contrib.itertools import product
import copy

def load_config(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def update_exp_group(config, group, iteration):
    '''
    Updates an experiment group in a config file for 1 treatment
    '''
    for key, value in config.items():
        if type(value) is dict:
            if 'group' in value.keys():
                if value['group'] == group:
                    config[key] = value['values'][iteration]
            else:
                config[key] = update_exp_group(value, group, iteration)
    return config

def main(args):
    config_file_list = glob(args.config_file)

    if args.save_images and os.path.isdir(args.save_location):
        raise Exception(f'Save location already exists: \"{args.save_location}\"! Please specify a new save location.')

    config = load_config(config_file_list[0])
    config_flat = pd.json_normalize(config, sep='.')

    results_columns = []
    final_results = None
    for config_num in tqdm(range(len(config_file_list)), desc='Config Files'):
        config_file = config_file_list[config_num]
        #print(f'------------- RUNNING CONFIG FILE: {config_file} -------------')
        
        config = load_config(config_file)
        config_flat = pd.json_normalize(config, sep='.')

        # Get a list of all experiment groups
        exp_groups = []
        for key, value in config_flat.items():
            if key.split('.')[-1] == 'group':
                if value.iloc[0] not in [x[0] for x in exp_groups]:
                    exp_groups.append((value.iloc[0], list(range(len(config_flat['.'.join(key.split('.')[:-1] + ['values'])].iloc[0])))))

        # Create full factorial expeirment
        for exp_idx, combination in enumerate(product(*[x[1] for x in exp_groups], desc='Experiment Iteration', leave=False)):
            # Copy original config
            config_exp = copy.deepcopy(config)

            # Loop through experiment groups and update config
            for group_idx, group in enumerate([x[0] for x in exp_groups]):
                config_exp = update_exp_group(config_exp, group, combination[group_idx])

            config_exp_flat = pd.json_normalize(config_exp, sep='.')

            save_location = os.path.join(args.save_location, os.path.basename(config_file.split('.')[0]), str(exp_idx))
            
            if args.save_images:
                os.makedirs(save_location)

            # Create Environment
            env = MDSEnv(config_exp)
            env.reset(config_exp['seed'])

            # Initialize actor
            actors = {}
            for nation in ['US', 'Japan']:
                if config_exp[nation]['actor_type'] == 'random':
                    actors[nation] = actor(env.action_space, env.threat_platforms, env.defended_assets, config_exp['placement_lat_band'], config_exp['placement_lng_band'])
                elif config_exp[nation]['actor_type'] == 'in_between':
                    actors[nation] = actor_in_between(env.action_space, env.threat_platforms, env.defended_assets, config_exp['placement_lat_band'], config_exp['placement_lng_band'])
                elif config_exp[nation]['actor_type'] == 'near_defense':
                    actors[nation] = actor_near_defense(env.action_space, env.threat_platforms, env.defended_assets, config_exp['placement_lat_band'], config_exp['placement_lng_band'])
                elif config_exp[nation]['actor_type'] == 'spaced_out':
                    actors[nation] = actor_spaced_out(env.action_space, env.threat_platforms, env.defended_assets, config_exp['placement_lat_band'], config_exp['placement_lng_band'])
                elif config_exp[nation]['actor_type'] == 'in_between_and_near_defense':
                    actors[nation] = actor_in_between_and_near_defense(env.action_space, env.threat_platforms, env.defended_assets, config_exp['placement_lat_band'], config_exp['placement_lng_band'])
                else:
                    raise Exception(f"Invalid actor type selected: {config_exp['actor_type']}")

            # Step through environment
            total_reward = 0
            for t in tqdm(range(config_exp['max_time_steps']), desc='Time Step', leave=False):
                #print(f'Current time step: {t}')

                for nation, actor_obj in actors.items():
                    action = actor_obj.get_action()
                    obs, reward, done, log = env.step(action, nation)
                    total_reward += reward

                # Render plot each step
                if args.render or args.save_images:
                    env.render(mode=args.render_mode)
                    
                    if args.save_images:
                        plt.savefig(os.path.join(save_location, f'{t}.png'))

                if done:
                    break

            #print(f'Total reward: {total_reward}')
            if final_results is None:
                results_columns = ['config_file', 'total_reward', 'reward_perc', 'US_budget'] + config_exp_flat.columns.values.tolist()
                final_results = pd.DataFrame([[config_file, total_reward, log['reward_perc'], log['US_budget']] + config_exp_flat.iloc[0].to_list()], columns=results_columns)
            else:
                temp_results = pd.DataFrame([[config_file, total_reward, log['reward_perc'], log['US_budget']] + config_exp_flat.iloc[0].to_list()], columns=results_columns)
                final_results = final_results.append(temp_results, ignore_index=True)

    print(final_results)
    final_results.set_index('config_file').to_csv('final_results.csv')

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

    #warnings.filterwarnings('ignore', '.*Shapely 2.*')
    warnings.filterwarnings('ignore')

    main(args)