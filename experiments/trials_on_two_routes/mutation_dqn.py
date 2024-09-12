import matplotlib.pyplot as plt
import os
import pandas as pd
from tensordict.nn import TensorDictModule, TensorDictSequential
import torch
from torchrl.collectors import SyncDataCollector
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
from torchrl.modules import MLP, QValueActor
from torchrl.data import CompositeSpec
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.envs.transforms import RenameTransform
from torchrl.modules.tensordict_module import QValueModule
from torchrl.trainers import (
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)
import wandb
import json
import time
from tqdm import tqdm
import sys
from keychain import Keychain as kc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environment.environment import TrafficEnvironment
from services.plotter import Plotter
from utilities import get_params

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


##################### Hyperparameters #####################

# Devices
device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)
vmas_device = device  # The device where the simulator is run

# Sampling
frames_per_batch = 200  # Number of team frames collected per training iteration
n_iters = 10  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 100  # Number of optimization steps per training iteration
minibatch_size = 2  # Size of the mini-batches in each optimization step
lr = 3e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# DQN
gamma = 0.99  # discount factor
hard_update_freq = 10

##################### Environment Creation #####################
params = get_params(kc.PARAMS_PATH)

env = TrafficEnvironment(params[kc.RUNNER], params[kc.ENVIRONMENT], params[kc.SIMULATOR], params[kc.AGENT_GEN], params[kc.AGENTS], params[kc.PHASE])

env.start()

##################### Human Learning #####################
num_episodes = 200

for episode in range(num_episodes):
    env.step()


##################### Mutation #####################
env.mutation()


################## Machine Learning ################
env = PettingZooWrapper(
    env=env,
    use_mask=True,
    group_map=None,
    categorical_actions=True,
    done_on_any = False,
    device = device
)

env = TransformedEnv(
    env,
    RewardSum(
        in_keys=env.reward_keys,
        reset_keys=["_reset"] * len(env.group_map.keys()),
    ),
    device = device
)

reset_td = env.reset()

##################### Policy network #####################
modules = {}
for group, agents in env.group_map.items():
    share_parameters_policy = False 

    mlp = MultiAgentMLP(
        n_agent_inputs =env.observation_spec[group, "observation"].shape[-1],  
        n_agent_outputs = env.full_action_spec[group, "action"].space.n, 
        n_agents = len(agents),
        centralised=False,  
        share_params = share_parameters_policy,
        device = device,
        depth = 4,
        num_cells = 64,
        activation_class=torch.nn.ReLU,
    )

    module = TensorDictModule(mlp, 
                              in_keys=[(group, "observation")],
                              out_keys=[(group,"action_value")],
    )

    modules[group] = module

q_value_modules = {}

for group, agents in env.group_map.items():

    q_value_module = QValueModule(
            action_value_key=(group, "action_value"),
            out_keys=[
                (group, "action"),
                (group, "action_value"),
                (group, "chosen_action_value"),
            ],
            spec=env.full_action_spec[group, "action"],
            action_space=None,
        )

    q_value_modules[group] = q_value_module

policy = TensorDictSequential(*modules.values(), *q_value_modules.values())

##################### Greedy module #####################
greedy_module = {}

for group, agents in env.group_map.items():

    greedy_module[group] = EGreedyModule(
        action_key = (group, "action"),
        spec=env.full_action_spec[group, "action"],
    )


# Incorporate the greedy module inside the policy.
col_policy = {}

for group, agents in env.group_map.items():
    col_policy[group] = TensorDictSequential(policy, greedy_module[group])

col_policies = TensorDictSequential(*col_policy.values())


##################### Collector #####################
collector = SyncDataCollector(
    env,
    col_policies,
    device=device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    reset_at_each_iter=False,
    total_frames=total_frames,
)

##################### Replay Buffer #####################
replay_buffers = {}
for group, _agents in env.group_map.items():
    replay_buffers[group] = ReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, device=device
        ), 
        batch_size=minibatch_size, 
    )

replay_buffer = replay_buffers[group]



##################### DQN loss function #####################
losses = {}
optimizers = {}
target_net_updaters = {}


for group, _agents in env.group_map.items():
    loss_module = DQNLoss(
        value_network=col_policies,
        loss_function="l2",
        double_dqn = False,
        delay_value=True,
        action_space = "categorical"
    )

    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=(group, "reward"),  
        action_value=(group, "action_value"),
        action=(group, "action"), 
        done=(group, "done"),
        terminated=(group, "terminated"),
        value=(group, "chosen_action_value"),
    )

    loss_module.make_value_estimator(gamma=gamma)

    target_net_updaters[group] = SoftUpdate(
        loss_module, eps=0.98
    )    

    losses[group] = loss_module

    optimizer = torch.optim.Adam(loss_module.parameters(), lr)
    
    optimizers[group] = optimizer

# Access loss module for the first group for example
group = next(iter(env.group_map))


q_losses_loop = {group: [] for group in env.group_map.keys()}
q_values = {group: [] for group in env.group_map.keys()}


##################### Create the logger #####################
"""wandb.login()

logger = None

exp_name = generate_exp_name("DQN", f"TrafficEnv")
logger = get_logger(
    "wandb",
    logger_name="dqn",
    experiment_name=exp_name,
    wandb_kwargs={
        "project": "2_machines_mutation",
    },
)"""


collected_frames = 0
start_time = time.time()
num_updates = 5
batch_size = 10
test_interval = 5
max_grad = 1
num_test_episodes = 5
frames_per_batch = frames_per_batch
init_random_frames = 5
n_optim = 8
q_losses = torch.zeros(num_updates, device=device)
sampling_start = time.time()


##################### Training loop #####################
pbar = tqdm(total=n_iters)
for i, tensordict_data in enumerate(collector):

    for group, _agents in env.group_map.items():
        tensordict_data.set(
            ("next", group, "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", group, "reward"))),  # Adjust index to start from 0
        )
        tensordict_data.set(
            ("next", group, "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", group, "reward"))),  # Adjust index to start from 0
        )

    log_info = {}
    sampling_time = time.time() - sampling_start

    data = tensordict_data.reshape(-1)
    current_frames = data.numel()
    collected_frames += current_frames

    for group, agents in env.group_map.items():
        replay_buffers[group].extend(data)
        greedy_module[group].step(current_frames)

    # Get and log training rewards and episode lengths
    
        episode_rewards = data["next", group, "episode_reward"][data["next", group, "done"]]
        if len(episode_rewards) > 0:
            episode_reward_mean = episode_rewards.mean().item()
            #episode_length = data["next", group, "step_count"][data["next", group, "done"]]
            #episode_length_mean = episode_length.sum().item() / len(episode_length)
            """log_info.update(
                {
                    f"train/episode_reward_{group}": episode_reward_mean,
                    #"train/episode_length": episode_length_mean,
                }
            )"""

        """if collected_frames < init_random_frames:
            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, step=collected_frames)
            continue"""


    # optimization steps
    training_start = time.time()
    for group, agent in env.group_map.items():
        for _ in range(frames_per_batch // minibatch_size):

            sampled_tensordict = replay_buffers[group].sample()
            sampled_tensordict = sampled_tensordict.to(device)

            loss_td = losses[group](sampled_tensordict)
            q_loss = loss_td["loss"]

            q_losses_loop[group].append(q_loss)
            ## One qvalue saved for each agent, each time it takes an action
            q_values[group].append((data[group, "action_value"] * data[group, "action"]).sum().item()/ frames_per_batch)

            optimizer.zero_grad()
            q_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(losses[group].parameters()), max_norm=max_grad
            )
            
            optimizers[group].step()
            target_net_updaters[group].step()

            training_time = time.time() - training_start

            # Get and log q-values, loss, epsilon, sampling time and training time
            """log_info.update(
                {
                    f"train/q_values_{group}": (data[group, "action_value"] * data[group, "action"]).sum().item()
                    / frames_per_batch,
                    f"train/q_loss_{group}": torch.stack(q_losses_loop[group]).mean().item(),
                    f"train/epsilon_{group}": greedy_module[group].eps,
                    "train/sampling_time": sampling_time,
                    "train/training_time": training_time,
                }
            )"""

            """if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, step=collected_frames)"""

            
            # update weights of the inference policy
            collector.update_policy_weights_()
            sampling_start = time.time()
    pbar.update()


collector.shutdown()
end_time = time.time()
execution_time = end_time - start_time
print(f"Training took {execution_time:.2f} seconds to finish")

q_losses_file = kc.RECORDS_FOLDER + '/q_losses_loop.json'
# loss is a tensor so I transform it to a list
q_losses_loop = {group: [tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]
        for group, tensors in q_losses_loop.items()}
with open(q_losses_file, 'w') as f:
    json.dump(q_losses_loop, f)

q_values_file = kc.RECORDS_FOLDER + '/q_values_loop.json'
q_values = {group: [tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]
        for group, tensors in q_values.items()}
with open(q_values_file, 'w') as f:
    json.dump(q_values, f)

from services import plotter
plotter(params[kc.PLOTTER])
