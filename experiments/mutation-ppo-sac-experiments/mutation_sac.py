import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from tensordict.nn import TensorDictModule, TensorDictSequential
import torch
from torchrl.collectors import SyncDataCollector
from torchrl._utils import logger as torchrl_logger
from torch.distributions import Categorical
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import TransformedEnv, RewardSum
from torchrl.envs.utils import check_env_specs
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.data import TensorDictReplayBuffer
from torchrl.objectives import DiscreteSACLoss
from torchrl.data.replay_buffers import LazyMemmapStorage
from torchrl.modules.tensordict_module import ValueOperator
from torchrl.modules import MLP, QValueActor
from torchrl.data import CompositeSpec
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.envs.transforms import RenameTransform
from torchrl.modules.tensordict_module import QValueModule
from torchrl.modules import DuelingCnnDQNet
from torchrl.modules.models.models import DuelingMlpDQNet
from torch import nn
from torchrl.trainers import (
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)
import wandb
import time
from tqdm import tqdm
import sys

current_dir = os.getcwd()
parent_of_parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_of_parent_dir)

from environment import TrafficEnvironment
from keychain import Keychain as kc
from services.plotter import Plotter
from utilities import get_params

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

""" Hyperparameters specification """
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
vmas_device = device  # The device where the simulator is run

# Sampling
frames_per_batch = 20  # Number of team frames collected per training iteration
n_iters = 40  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 100  # Number of optimization steps per training iteration
minibatch_size = 2  # Size of the mini-batches in each optimization step
lr = 3e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# DQN
gamma = 0.99  # discount factor
hard_update_freq = 10
tau = 0.02
init_bias = 2.0 #To speed up learning, we set the bias of the last layer of our value network to a predefined value (this is not mandatory)


def policy_network(env):

    policy_modules = {}
    for group, agents in env.group_map.items():
        share_parameters_policy = False  

        policy_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec[group, "observation"].shape[-1], 
            n_agent_outputs = env.full_action_spec[group, "action"].space.n, 
            n_agents=len(agents),  
            centralised=False,  
            share_params=share_parameters_policy,
            device=device,
            depth=4,
            num_cells=64,
            activation_class=torch.nn.Tanh,
        )

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[(group, "observation")],
            out_keys=[(group, "logits")],
        )
        policy_modules[group] = policy_module


    policies = {}

    for group, _agents in env.group_map.items():

        policy = ProbabilisticActor(
            module=policy_modules[group],
            spec=env.full_action_spec[group, "action"],
            in_keys=[(group, "logits")],
            out_keys=[(group, "action")],
            distribution_class=Categorical,
            return_log_prob=True,
            log_prob_key=(group, "sample_log_prob"),
        )
        
        policies[group] = policy

    return policies

    
def critic_network(env):

    critic_modules = {}

    for group, agents in env.group_map.items():
        share_parameters_critic = False
        mappo = False 

        critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec[group, "observation"].shape[-1],
            n_agent_outputs = env.full_action_spec[group, "action"].space.n, 
            n_agents = len(agents),
            centralised=mappo,
            share_params=share_parameters_critic,
            device=device,
            depth=4,
            num_cells=128,
            activation_class=torch.nn.Tanh,
        )

        value_module = ValueOperator(
            module=critic_net,
            in_keys=[(group, "observation")],
            out_keys=[(group, "action_value")],
        )
        critic_modules[group] = value_module

    return critic_modules


def transform_environment(env):
    ### Transform env
    out_keys = []

    for group, agents in env.group_map.items():
        out_keys.append((group, "episode_reward"))

    env = TransformedEnv(
        env,
        RewardSum(
            in_keys=env.reward_keys,
            reset_keys=["_reset"] * len(env.group_map.keys()),
            out_keys = out_keys
        ),
    )

    return env

def collector(env):

    collector = SyncDataCollector(
        env,
        policy,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        reset_at_each_iter=True,
        total_frames=total_frames,
    )

    return collector

def replay_buffer(env):

    replay_buffers = {}

    for group, _agents in env.group_map.items():

        replay_buffers[group] = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                10 * frames_per_batch, device=device
            ),  
            sampler=SamplerWithoutReplacement(),
            batch_size=10*minibatch_size, 
        )

    return replay_buffers

def sac_loss_function(env, policies, critic_modules):
    losses = {}
    optimizers = {}
    target_net_updaters = {}

    for group, _agents in env.group_map.items():
        
        loss_module = DiscreteSACLoss(
            actor_network=policies[group],
            qvalue_network=critic_modules[group],
            delay_qvalue=True, ## Whether to separate the target Q value networks from the Q value networks used for data collection.
            num_actions=env.action_spec[group]['action'].space.n,
            action_space=env.action_spec[group]['action'] ### changed this - don't know if it's correct
        )
        loss_module.set_keys(  # We have to tell the loss where to find the keys
            reward=(group, "reward"),  
            action_value=(group, "action_value"),
            action=(group, "action"), 
            done=(group, "done"),
            terminated=(group, "terminated"),
        )

        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)

        target_net_updaters[group] = SoftUpdate(loss_module, eps=1 - tau)


        losses[group] = loss_module

        optimizers[group] = torch.optim.Adam(loss_module.parameters(), lr)

    return optimizers, losses, target_net_updaters



""" Environment creation """

params = get_params(kc.PARAMS_PATH)

env = TrafficEnvironment(params[kc.RUNNER], params[kc.ENVIRONMENT], params[kc.SIMULATOR], params[kc.AGENT_GEN], params[kc.AGENTS], params[kc.PHASE])

env.start()

""" Human learning """

num_episodes = 100

for episode in range(num_episodes):
    env.step()


""" Mutation """
env.mutation()


""" Machine learning """
env = PettingZooWrapper(
    env=env,
    use_mask=True,
    group_map=None,
    categorical_actions=True,
    done_on_any = False
)


""" Check env """
check_env_specs(env)

reset_td = env.reset()


env = transform_environment(env)

## Actor network
policies =  policy_network(env)
policy = TensorDictSequential(*policies.values())

## Critic network
critic_modules = critic_network(env)

## Collector
collector = collector(env)

## Replay buffer
replay_buffers = replay_buffer(env)

## Loss function
optimizers, losses, target_net_updaters = sac_loss_function(env, policies, critic_modules)


## Training loop
pbar = tqdm(
    total=n_iters,
    desc=", ".join(
        [f"episode_reward_mean_{group} = 0" for group in env.group_map.keys()]
    ),
)
sampling_start = time.time()
total_time = 0
episode_reward_mean_map = {group: [] for group in env.group_map.keys()}


for i, tensordict_data in enumerate(collector):
    torchrl_logger.info(f"\nIteration {i}")

    sampling_time = time.time() - sampling_start

    current_frames = tensordict_data.numel()
    total_frames += current_frames

    for group, _agents in env.group_map.items():
        data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
        replay_buffers[group].extend(data_view)

    training_tds = []
    training_start = time.time()
    for _ in range(num_epochs):
        for group, _agents in env.group_map.items():
            for _ in range(frames_per_batch // minibatch_size):

                subdata = replay_buffers[group].sample()

                loss_vals = losses[group](subdata)
                training_tds.append(loss_vals.detach())

                loss_value = (
                    loss_vals["loss_actor"]
                    + loss_vals["loss_alpha"]
                    + loss_vals["loss_qvalue"]
                )

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    losses[group].parameters(), max_grad_norm
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optimizers[group].step()
                optimizers[group].zero_grad()
                
                target_net_updaters[group].step()

    collector.update_policy_weights_()

    training_time = time.time() - training_start

    iteration_time = sampling_time + training_time
    total_time += iteration_time
    training_tds = torch.stack(training_tds)

    for group, _agents in env.group_map.items():
        done = tensordict_data.get(("next", group, "done"))  # Get done status for the group

        episode_reward_mean = (
            tensordict_data.get(("next", group, "reward"))[
                tensordict_data.get(("next", group, "done"))
            ]
            .mean()
            .item()
        )

        episode_reward_mean_map[group].append(episode_reward_mean)

    pbar.set_description(
        ", ".join(
            [
                f"episode_reward_mean_{group} = {episode_reward_mean_map[group][-1]}"
                for group in env.group_map.keys()
            ]
        ),
        refresh=False,
    )
    pbar.update()


## Save mean episode reward
with open('episode_reward_mean.pkl', 'wb') as f:
    pickle.dump(episode_reward_mean_map, f)

## Save training_tds
torch.save(training_tds, 'my_tensor.pt')