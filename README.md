<img src="docs/_static/logo.png" align="right" width="20%"/>

# RouteRL

[![Tutorials Tests](https://github.com/COeXISTENCE-PROJECT/RouteRL/actions/workflows/test_tutorials.yml/badge.svg)](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/notebooks)
[![Online Documentation](https://github.com/COeXISTENCE-PROJECT/RouteRL/actions/workflows/documentation.yml/badge.svg)](https://coexistence-project.github.io/RouteRL/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/LICENSE.txt)

<!-- start intro -->

RouteRL is a multi-agent reinforcement learning environment for urban route choice that simulates the coexistence of human drivers and Automated Vehicles (AVs) in city networks. 

- The main class is [TrafficEnvironment](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/routerl/environment/environment.py) and is a [PettingZoo](https://pettingzoo.farama.org/index.html) AEC API environment.
- There are two types of agents in the environment and are both represented by the [BaseAgent](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/3d2ca55e4474eee062f161c42f47a212b3936377/routerl/environment/agent.py#L14) class.
  - Human drivers are simulated using human route-choice behavior from transportation research.
  - Automated vehicles (AVs) are the RL agents that aim to optimize their routes and learn the most efficient paths.
- It is compatible with popular RL libraries such as [TorchRL](https://pytorch.org/rl/stable/tutorials/torchrl_demo.html).

<!-- end intro -->

For more details, check the documentation [online](https://coexistence-project.github.io/RouteRL/).

## RouteRL usage and functionalities at glance

```python
env = TrafficEnvironment(seed=42, **env_params) # initialize the traffic environment

env.start() # start the connection with SUMO

# Human learning 
for episode in range(human_learning_episodes): 
    env.step()

env.mutation() # some human agents transition to AV agents

# PettingZoo environment wrapper
group = {'agents': [str(machine.id) for machine in env.machine_agents]}

env = PettingZooWrapper(
    env=env,
    use_mask=True,
    categorical_actions=True,
    done_on_any = False,
    group_map=group,
    device=device
)

# Policy network
net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=env.action_spec.space.n,
        n_agents=env.n_agents,
        centralised=False,
        share_params=False,
        device=device,
        depth=mlp_depth,
        num_cells=mlp_cells,
        activation_class=nn.ReLU,
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

# Collector
collector = SyncDataCollector(
        env,
        qnet,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )

# Replay buffer
replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(memory_size, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

# DQN loss function
loss_module = DQNLoss(qnet, delay_value=True)

loss_module.set_keys(
        action_value=("agents", "action_value"),
        action=env.action_key,
        value=("agents", "chosen_action_value"),
        reward=env.reward_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
)

loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
target_net_updater = SoftUpdate(loss_module, eps=eps)

optim = torch.optim.Adam(loss_module.parameters(), lr)

# Training loop

for i, tensordict_data in tqdm(enumerate(collector), total=n_iters, desc="Training"):
    
    current_frames = tensordict_data.numel()
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

            total_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            training_tds[-1].set("grad_norm", total_norm.mean())

            optim.step()
            optim.zero_grad()
        target_net_updater.step()

    qnet_explore[1].step(frames=current_frames)  # Update exploration annealing
    collector.update_policy_weights_()
    
    training_tds = torch.stack(training_tds) 

collector.shutdown()

# Testing phase

num_episodes = 100
for episode in range(num_episodes):
    env.rollout(len(env.machine_agents), policy=qnet)

# Plot the results 
env.plot_results()

# Stop the connection with SUMO
env.stop_simulation()

```


## Documentation


* [Tutorials](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials):
  * [Quickstart](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials/1_Quickstart_TraffficEnvironment_Introduction).
  * [Medium network and AVs behaviors](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials/2_MediumNetwork_AVsBehaviors_TorchRL_CollaborativeAlgorithms).
  * [Big network and independent AV agents](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials/3_BiggerNetwork_IndependentAgents).
  * [Different MARL algorithms](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials/4_CsomorNetwork_DifferentMARLAlgorithms).


## Installation

<!-- start installation -->

- **Prerequisite**: Make sure you have SUMO installed in your system. This procedure should be carried out separately, by following the instructions provided [here](https://sumo.dlr.de/docs/Installing/index.html).
- **Option 1**: Install the latest stable version from PyPI:  
  ```
    pip install routerl
  ```
- **Option 2**: Clone this repository for latest version, and manually install its dependencies: 
  ```
    git clone https://github.com/COeXISTENCE-PROJECT/RouteRL.git
    cd RouteRL
    pip install -r requirements.txt
  ```
 
<!-- end installation -->  


----------------

## Citation

```bibtex
@misc{RouteRL,
  author      = {Ahmet Onur Akman, Anastasia Psarou, Łukasz Gorczyca, Zoltan\a{'}n Görgy Varga, Grzegorz Jamróz, 
  Rafał Kucharski},
  title       = {RouteRL},
  year        = {2025},
  url         = {https://github.com/COeXISTENCE-PROJECT/RouteRL},
  publisher   = {GitHub},
  journal     = {GitHub repository}
}
```

