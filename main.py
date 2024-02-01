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
import glob

torch, nn = try_import_torch()



confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_json(kc.PARAMS_PATH)

"""class TorchMaskedActions(DQNTorchModel):
    def __init__(
        self,
        obs_space: Box,
        action_space: gym.spaces.Discrete,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        print("\n\n\n\n inside init \n\n\n\n")
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )

        print("\n\n\n\n after DQNTorchModel \n\n\n\n")

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

        print("\n\n before completing forward \n\n")

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()"""

### CleanRL
class Agent(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size=128):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(obs_size, 32)),
            nn.ReLU(),
            self._layer_init(nn.Linear(32, 64)),
            nn.ReLU(),
            self._layer_init(nn.Linear(64, 128)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(128, n_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(128, 1))


    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):      
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
        



def main():

    env = TrafficEnvironment(params[kc.SIMULATION_PARAMETERS])

    #agents = create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS], env.calculate_free_flow_times())

    parallel_api_test(env, num_cycles=1_000_000)


    ##### CleanRL
    device = "cpu"
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape


    #LEARNER SETUP 
    agent = Agent(0, num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

    print(agent, optimizer)

    #ALGO LOGIC: EPISODE STORAGE
    end_step = 0
    max_cycles = 125
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    print("\nGoing towards training logic\n")
    #TRAINING LOGIC
    # train for n number of episodes
    total_episodes = 2
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=None)
            # reset the episodic return
            total_episodic_return = 0

            print("\n Going inside the loop\n")
            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                obs = next_obs #batchify_obs(next_obs, device)

                # get action from the agent
                print(obs)
                x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
                actions, logprobs, _, values = agent.get_action_and_value(x)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(actions)

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = rewards
                rb_terms[step] = terms
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()
                print("total_episodic return is: ", total_episodic_return)

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break



    """ray.init()


    alg_name = "DQN"
    ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)

    def env_creator(args): ### add here the env config
        env = TrafficEnvironment(params[kc.SIMULATION_PARAMETERS])
        return env
    

    register_env("TrafficEnvironment", lambda config: ParallelPettingZooEnv(env_creator(config)))

    config = (
        PPOConfig()
        .environment(env="TrafficEnvironment", disable_env_checking=True)
        .rollouts(num_rollout_workers=2, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.95,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 100000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + "TrafficEnvironment",
        config=config.to_dict(),
    )"""


    """config = (
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
                }
            }
        )
        .multi_agent(
            policies={              # observation space                         # action space
                "pol1": PolicySpec(None, gym.spaces.Box(low=0, high=1, shape=(1,), dtype=float), gym.spaces.Discrete(3), {"lr": 0.0001}),
                "pol2": PolicySpec(None, gym.spaces.Box(low=0, high=1, shape=(1,), dtype=float), gym.spaces.Discrete(3), {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id), # each policy maps to the specific agent
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
    )"""

    
    

main()