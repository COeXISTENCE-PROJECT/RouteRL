from environment import TrafficEnvironment
from MultiAgentWrapper import MultiAgentEnvWrapper

#import gymnasium as gym
from keychain import Keychain as kc
import os
from services import Trainer
from services import create_agent_objects
from services import confirm_env_variable
from services import get_json
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from pettingzoo.test import parallel_api_test
from pettingzoo.test import seed_test, parallel_seed_test
import numpy as np
import os

import ray
from gymnasium.spaces import Box, Discrete
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.tune.registry import register_env

from pettingzoo.classic import leduc_holdem_v4

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from torch import nn
import gymnasium as gym
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.test import test_save_obs
from torch.distributions.categorical import Categorical
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import supersuit as ss

confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_json(kc.PARAMS_PATH)

### Stable baselines
def train_butterfly_supersuit(env, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in a cooperative Parallel environment

    env.reset(seed=seed)

    print(f"[SUCCESS] Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, 4, num_cpus=0, base_class="stable_baselines3")

    """model = PPO(
        "MlpPolicy",
        env,
        verbose = 3,
        learning_rate = 1e-3,
        n_steps = 10,
        batch_size=10
    )"""

    model = DQN(
        env=env,
        policy="MlpPolicy",
        #tensorboard_log="./board/",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0,
        verbose=1,
    )

    model.learn(total_timesteps=10)

    print(f"[SUCCESS] Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

def main():

    env = TrafficEnvironment(params[kc.SIMULATION_PARAMETERS])
    
    parallel_api_test(env, num_cycles=1_000_000)
    print("\n[SUCCESS] Passed parallel_api_test\n")

    env_kwargs = {}
    train_butterfly_supersuit(env, steps=100, seed=0, **env_kwargs)    

    print("Going to eval")    

main()