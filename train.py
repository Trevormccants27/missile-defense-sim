from ray.tune.registry import register_env
import ray
import ray.rllib.agents.ppo as ppo
import yaml

from mds_env import MDSEnv

def load_config(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def main():
    ray.init(ignore_reinit_error=True)

    env_config = load_config('config.yml')
    register_env('mds_env', lambda trash: MDSEnv(env_config))

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config['model']['conv_filters'] = [[16, [3, 3], 4], [32, [3, 3], 4]]
    ppo_config['vf_clip_param'] = 10000
    agent = ppo.PPOTrainer(ppo_config, env='mds_env')

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 5

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save('./checkpoint')

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))


if __name__ == "__main__":
    main()