import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))

import torch
from tqdm import tqdm
import numpy as np

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import TransformedEnv, RewardSum
from torchrl.envs.utils import check_env_specs
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import MultiAgentMLP, QMixer
from torchrl.objectives import SoftUpdate, ValueEstimators
from torchrl.objectives.multiagent.qmixer import QMixerLoss
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))


from routerl import TrafficEnvironment

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Devices
device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("device is: ", device)

# Sampling
frames_per_batch = 1000  # Number of team frames collected per training iteration
n_iters = 400  # Number of sampling and training iterations - the episodes the plotter plots
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 1  # Number of optimization steps per training iteration
minibatch_size = 2  # Size of the mini-batches in each optimization step
lr = 3e-2  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients
memory_size = 1000  # Size of the replay buffer
tau =  0.005
gamma = 0.99  # discount factor

mlp_depth=2
mlp_num_cells=256

eps_greedy_init=0.3
eps_greedy_end=0

mixing_embed_dim = 32

human_learning_episodes = 100
new_machines_after_mutation = 40

# number of episodes the AV training will take
training_episodes = (frames_per_batch / new_machines_after_mutation) * n_iters

# Environment
env_params = {
    "agent_parameters" : {
        "new_machines_after_mutation": new_machines_after_mutation,

        "human_parameters" :
        {
            "model" : "gawron",

            "noise_weight_agent" : 0,
            "noise_weight_path" : 0.8,
            "noise_weight_day" : 0.2,

            "beta" : -1,
            "beta_k_i_variability" : 0.1,
            "epsilon_i_variability" : 0.1,
            "epsilon_k_i_variability" : 0.1,
            "epsilon_k_i_t_variability" : 0.1,

            "greedy" : 0.1,
            "gamma_c" : 0.0,
            "gamma_u" : 0.0,
            "remember" : 1,

            "alpha_zero" : 0.8,
            "alphas" : [0.2]  
        },
        "machine_parameters" :
        {
            "behavior" : "selfish",
        }
    },
    "simulator_parameters" : {
        "network_name" : "ingolstadt",
        "sumo_type" : "sumo",
    },  
    "plotter_parameters" : {
        "phases" : [0, human_learning_episodes, int(training_episodes) + human_learning_episodes],
        "smooth_by" : 50,
        "plot_choices" : "basic",
        "phase_names" : [
            "Human learning", 
            "Mutation - Machine learning",
            "Testing phase"
        ],
        "records_folder" : "qmix_mutation_1"
    },
    "path_generation_parameters":
    {
        "origins" : ['315358244', '10425609#0', '24608844', '315358250#0', '-306240162#1', '201201950#1', '201201953#4', '272042143', '266565295#5', '22716069#0', '24634507', '128361109#1', '315358242#0', '-447569998#1', '-54169231#2', '-10427692#1', '-32978638#0', '-18809673#6', '201950247#3', '-26677542#0', '26677213#0', '24599188#0.94', '176550249#3', '-24634413#2', '-160314345#5', '24634517#1', '201238726#0.117', '28319300#3', '-399835085#0', '-160314345#2', '168702040#1', '24634416', '-24634509#7', '-137454133#5', '-315358257#0.26', '-26677216#0', '286646456#1', '201201935#0', '-24634414#1', '23166741#2', '-24634510#5', '201238719#0', '24634513#0', '40888360#0', '24634510#1', '-224892361#8', '54169231#1', '315392062#0', '53396619#4', '-201950247#0', '-24634411', '176550249#4', '-24634505', '176550249#1', '28111977#8', '-201950263#6', '-24634413#0', '176550249#0', '129379925', '-28319300#2', '-22716549#6', '-201089423#1', '-25117391#2', '-201201945#0', '-136728347#4', '315358242#1', '22879845', '-25117391#1', '24634411', '-315358242#0', '-22724699#11', '26677214#0', '-26677213#5', '-315392062#4', '315358242#3', '23436553#4', '26676668', '-40888356#6', '-201238718#1', '24634517#6', '-24634506#0', '25117391#2', '-10427692#6', '-22690206#1', '-25145014#4', '-24634510#6', '286646456#0', '10427692#7', '-25187895#0', '-26677539', '201950247#7', '-24634517#16', '201950263#0', '201201950#3', '128361109#3', '23525483#7', '-18809672#6', '-24634417', '201201945#3', '-22716073', '25145014#0', '-24634514#1', '26677542#1', '-24634510#15', '24634414#5', '25190140#1', '-32124743', '23166741#5', '25117391#0', '-224892339#3', '-233675413#0', '28319300#0', '-201950263#10', '-24634517#3', '-201201945#6', '-40888351#2', '173203413#0', '32395327#0', '-54169231#0', '31860333#0', '24634417', '218647954', '-24634513#5', '201238724', '-170018165#3', '-201950259#6', '26677214#5', '24608845', '24634509#1', '-201238718#0', '18809672#7', '25149001#4', '201089423#0', '160314345#2', '-24634508', '24634517#13', '201950259#0', '160314345#3', '-25117417#1', '-224892361#6', '26677540#0', '224251774#1', '24634511', '26677540#1', '-40888351#6', '24634413#0', '-201963533#5', '-36962701#1', '23525483#1', '-201950259#3', '-41203916#3', '201201945#1', '-201950247#5', '-40888443', '24634516#1', '-24634510#0', '18813598#8', '170018165#2.175', '24693977#1', '-26677417#0', '-22690206#2', '24634510#0', '-201963533#1', '201950247#8', '24608846#0', '26677216#1', '201201945#2', '315358253#2', '-26677541', '18813598#7', '170425366#0', '32124637#1', '-201238724', '22690205#3', '22724699#3', '201201953#2', '-173203413#7', '-224892361#1', '24634506#0', '-137454133#0', '-22724699#2', '32021112#0', '201238726#0', '-24634511', '-18813598#7', '-28111977#9', '26677214#3', '25145012#5', '-36962701#0', '168702039#1', '10427692#6', '-37681424#1', '224892361#3', '-26677216#4', '315358246', '-54169231#1', '-138300620#6', '40888354#0', '653473569#3', '-315358253#1', '201201950#4', '-24634507', '-24634510#18', '26677541', '26677542#0', '26677216#0', '24634506#2', '-201963533#1.145', '-137454133#1', '-26677542#1', '-32999434#1', '128906555', '-315358251#0', '-315358255#4', '-22690205#1', '-25145013#0'],
        "destinations" : ['-25149001#4', '-128361102#1', '32978638#0', '386687235', '399835085#1', '-266565295#5', '24634414#1', '-24634415', '24634517#13', '201238729#3', '306240162#0', '-26677539', '447569997#1', '40888354#0', '315358242#0', '-24634506#2', '30482615#1', '-10427692#3', '24634510#6', '-24634505', '272042145', '-201238724', '-83304175#2', '26677541', '201238718#0', '-25145014#0', '24634509#1', '-315358242#1', '-315358242#0', '402600768#1', '28319300#3', '-201963533#4', '24608845', '-25190140#1', '137454133#2', '201201950#1', '40888354#1', '40888356#0', '138300620#0', '-201238718#0', '-22724699#1', '-32395288#1', '-218647954', '22724699#7', '26677214#3', '24634414#2', '-160314346#0', '201201953#10', '28111977#9', '-25145011#0', '-170018165#0', '23166741#5', '4942376', '24634506#2', '31860333#1', '176550249#4', '-315358258', '26642363', '224892361#3', '-160314345#0', '26677214#4', '-26677416#1', '28319300#0', '-25187893', '201238726#0', '24634411', '-32395361#1', '201950250#1', '-25145012#3', '-26677213#5', '22716549#0', '-315358252#1', '-24634413#2', '-393420106#1', '-24634521#1', '-201963533#1.145', '24634513#0', '-26677216#0', '-24693977#1', '-379510292', '-24634507', '-201963522#3', '-201950247#3', '-24634510#5', '-24634518#0', '176550249#3', '201950247#0', '24634413#1', '28111977#0', '24634511', '40888359', '170018165#2', '51857516#1', '32999110#0', '201201953#12', '-201201945#0', '-129379918#1', '-170018165#3', '24634510#1', '-26677542#1', '201950247#6', '-201238718#1', '53396619#3', '-201201945#0.78', '24634507', '-201950247#5', '-23436553#2', '-315358246', '-24634413#0', '-315392062#4', '653473569#1', '24634510#7', '201950263#0', '201950263#7', '25117417#1', '-22724699#2', '-22724699#11', '201963522#6', '-24634506#0', '-136436468#0', '-10427692#7', '-24634510#18', '-40888350', '201950259#0', '-201950259#6', '-24634513#5', '315392062#0', '201950250#0', '-22690205#3', '-26677540#1', '-138300620#6', '-201950263#6', '26677542#1', '-25187895#1', '-22690206#1', '160314346#0', '-53396619#2', '40888354#2', '-26677417#0', '-32999434#1', '-24634508', '-24634514#1', '24634517#1', '-201950247#7', '129379967#0', '315358257#0', '-315358248#1', '23525483#1', '-23525483#5', '286646456#1', '-22716549#6', '-224892361#6', '22724699#3', '136436468#1', '-129379922', '-24634414#5', '160314345#1', '204588664#0', '22724699#2', '315358245.27', '286646456#0', '160314345#2', '-173203413#7', '-315358257#2', '-286646456#0', '-26676668', '315358255#0', '23436553#2'],
        "number_of_paths" : 4,
        "beta" : -5,
        "num_samples" : 10,
        "visualize_paths" : True
    } 
}

env = TrafficEnvironment(seed=42, create_agents=False, create_paths=False, **env_params)

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

env.start()
env.reset()

for human in env.human_agents:

    inverse = 1 / np.array(human.initial_knowledge)
    invserse_normalized = inverse / inverse.sum()

    indices = np.arange(len(human.initial_knowledge))
    human.default_action = np.random.choice(indices, size=1, p=invserse_normalized)[0]
    
    print(human.default_action)

env.mutation()

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

machine_list = []
for machines in env.machine_agents:
    machine_list.append(str(machines.id))
      
group = {'agents': machine_list}

env = PettingZooWrapper(
    env=env,
    use_mask=True, # Whether to use the mask in the outputs. It is important for AEC environments to mask out non-acting agents.
    categorical_actions=True,
    done_on_any = False, # Whether the environment’s done keys are set by aggregating the agent keys using any() (when True) or all() (when False).
    group_map=group,
    device=device
)

print("env.group is: ", env.group_map, "\n\n")

env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)

check_env_specs(env)

reset_td = env.reset()

net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=env.action_spec.space.n,
        n_agents=env.n_agents,
        centralised=False,
        share_params=True,
        device=device,
        depth=mlp_depth,
        num_cells=mlp_num_cells,
        activation_class=nn.Tanh,
    )

module = TensorDictModule(
        net, in_keys=[("agents", "observation")], out_keys=[("agents", "action_value")]
)

value_module = QValueModule(
    action_value_key=("agents", "action_value"),
    out_keys=[
        env.action_key,
        ("agents", "action_value"),
        ("agents", "chosen_action_value"),
    ],
    spec=env.action_spec,
    action_space=None,
)

qnet = SafeSequential(module, value_module)

qnet_explore = TensorDictSequential(
    qnet,
    EGreedyModule(
        eps_init=eps_greedy_init,
        eps_end=eps_greedy_end,
        annealing_num_steps=int(total_frames * (1 / 2)), # Number of steps it will take for epsilon to reach the eps_end value
        action_key=env.action_key, # The key where the action can be found in the input tensordict.
        spec=env.action_spec,
    ),
)

mixer = TensorDictModule(
    module=QMixer(
        state_shape=env.observation_spec[
            "agents", "observation"
        ].shape,
        mixing_embed_dim=mixing_embed_dim,
        n_agents=env.n_agents,
        device=device,
    ),
    in_keys=[("agents", "chosen_action_value"), ("agents", "observation")],
    out_keys=["chosen_action_value"],
)

collector = SyncDataCollector(
        env,
        qnet_explore,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )

replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(memory_size, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

loss_module = QMixerLoss(qnet, mixer, delay_value=True)

loss_module.set_keys(
    action_value=("agents", "action_value"),
    local_value=("agents", "chosen_action_value"),
    global_value="chosen_action_value",
    action=env.action_key,
)

loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma) # The value estimator used for the loss computation
target_net_updater = SoftUpdate(loss_module, eps=1 - tau) # Technique used to update the target network

optim = torch.optim.Adam(loss_module.parameters(), lr)


for i, tensordict_data in tqdm(enumerate(collector), total=n_iters, desc="Training"):

    ## Generate the rollouts
    tensordict_data.set(
        ("next", "reward"), tensordict_data.get(("next", env.reward_key)).mean(-2)
    )
    del tensordict_data["next", env.reward_key]
    tensordict_data.set(
        ("next", "episode_reward"),
        tensordict_data.get(("next", "agents", "episode_reward")).mean(-2),
    )
    del tensordict_data["next", "agents", "episode_reward"]


    current_frames = tensordict_data.numel()
    total_frames += current_frames
    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view)
    

    training_tds = []

    ## Update the policies of the learning agents
    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)
            training_tds.append(loss_vals.detach())

            loss_value = loss_vals["loss"]

            loss_value.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )
            training_tds[-1].set("grad_norm", total_norm.mean())

            optim.step()
            optim.zero_grad()
            target_net_updater.step()

    qnet_explore[1].step(frames=current_frames)  # Update exploration annealing
    collector.update_policy_weights_()

    training_tds = torch.stack(training_tds) 

qnet_explore.eval() # set the policy into evaluation mode

num_episodes = 5
for episode in range(num_episodes):
    env.rollout(len(env.machine_agents), policy=qnet_explore)

env.plot_results()
env.stop_simulation()