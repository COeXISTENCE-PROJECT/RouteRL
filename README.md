<img src="docs/_static/logo.png" align="right" width="20%"/>

# RouteRL

![Notebook Tests](https://github.com/COeXISTENCE-PROJECT/RouteRL/actions/workflows/test_tutorials.yml/badge.svg)

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

# Human and AV agents interact with the environment and AVs are using a random policy
for episode in range(episodes): 
    print(f"\nStarting episode {episode + 1}")
    env.reset()
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # Policy action or random sampling
            action = env.action_space(agent).sample()
        print(f"Agent {agent} takes action: {action}")
        
        env.step(action)
        print(f"Agent {agent} has stepped, environment updated.\n")

```


## Documentation


* [Tutorials](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials):
  * [Quickstart](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials/1_Quickstart_TraffficEnvironment_Introduction).
  * [Medium network and AVs behaviors](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials/2_MediumNetwork_AVsBehaviors_TorchRL_CollaborativeAlgorithms).
  * [Big network and independent AV agents](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials/3_BiggerNetwork_IndependentAgents).
  * [Different MARL algorithms](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials/4_CsomorNetwork_DifferentMARLAlgorithms).




<!--# How to run on servers?

See [here](server_scripts/how_to.md).

# PettingZoo environment

<p float="left">
  <img src="images/multiple_humans_timesteps.png" alt="Image 1" width="480" />
  <img src="images/multiple_machines_timesteps.png" alt="Image 2"  width="300" />
</p>

# Training setting

## Number of agents
- 8 agents
- Humans: 4 | AVs: 4 
- Humans: Gawron | AVs: PPO / SAC
## AVs' objective
- **Selfish**: Minimize own travel time.
## Training episodes
- 10000 episodes, 3 phases
- Phase 1 (**Human Learning**) : Starts in episode 0
    - Humans: 8
    - Only humans learn.
- Phase 2 (**Mutation**) : Starts in episode 100
    - Humans: 4  AVs: 4 
    - Only machines learn.
## Training duration
- ~1.30 hours
## Hardware
- Anastasia's PC

<br><br><br>

# Results
#### *All plots smoothed by n=50*

## Travel times (in minutes)
![](readme_plots/travel_times.png)


## Distribution of Travel Times
![](readme_plots/tt_dist.png)


## Collected Mean Rewards
![](readme_plots/rewards.png)


## Mean Losses of DNNs of AVs 
#### (Throughout their learning)
![](readme_plots/losses.png)


## Simulation Timesteps
![](readme_plots/simulation_length.png)


## Picked Actions for OD Pairs
![](readme_plots/actions.png)


## Action Selection Shifts After Mutation
![](readme_plots/actions_shifts.png)-->
