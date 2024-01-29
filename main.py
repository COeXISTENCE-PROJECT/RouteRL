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
from torch import nn
import gymnasium as gym

torch, nn = try_import_torch()



confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_json(kc.PARAMS_PATH)

class TorchMaskedActions(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(
        self,
        obs_space: Box,
        action_space: gym.spaces.Discrete,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )

        obs_len = obs_space.shape[0] - action_space.n

        orig_obs_space = Box(
            low=0, high=1, shape=(1,), dtype=float
        )
        self.action_embed_model = TorchFC(
            orig_obs_space,
            action_space,
            action_space.n,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        print("\n\n\n\n\n input_dict", input_dict["obs"], "\n\n\n\n\n")
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model(
            {"obs": input_dict["obs"]}
        )
        # turns probit action mask into logit action mask
        inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

def main():

    env = TrafficEnvironment(params[kc.SIMULATION_PARAMETERS])

    #agents = create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS], env.calculate_free_flow_times())

    def env_creator():
        env = TrafficEnvironment(params[kc.SIMULATION_PARAMETERS])
        return env

    parallel_api_test(env, num_cycles=1_000_000)

    ray.init()




    alg_name = "DQN"
    ModelCatalog.register_custom_model("pa_model", TorchModelV2)

    def env_creator():
        env = TrafficEnvironment(params[kc.SIMULATION_PARAMETERS])
        return env

    register_env("TrafficEnvironment", lambda config: PettingZooEnv(env_creator()))

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = gym.spaces.Discrete(3)

    print("act_space is: ", act_space, "\n\n\n")

    config = (
        DQNConfig()
        .environment(env="TrafficEnvironment")
        .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
        .training(
            train_batch_size=200,
            hiddens=[],
            dueling=False,
            model={"custom_model": "pa_model"},
            optimizer={
            "adam": {
                "lr": 0.01,  # learning rate
                "momentum": 0.9,  # momentum (if applicable)
                # Add other optimizer parameters as needed
            }
        }
        )
        .multi_agent(
            policies={
                "player_0": (None, obs_space, act_space, {}),
                "player_1": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .debugging(
            log_level="DEBUG"
        )  # TODO: change to ERROR to match pistonball example
        .framework(framework="torch")
        .exploration(
            exploration_config={
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 0.1,
                "final_epsilon": 0.0,
                "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
            }
        )
    )

    config.environment(disable_env_checking=True)



    tune.run(
        alg_name,
        name="DQN",
        stop={"timesteps_total": 10000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        config=config.to_dict(),
    )

    

    
    

main()