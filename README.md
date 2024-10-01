# RouteRL

<!-- start intro -->
RouteRL provides a Multi-Agent Reinforcement Environment (MARL) for optimal route choice in different city networks. 

- The main class is [TrafficEnvironment](https://github.com/COeXISTENCE-PROJECT/Milestone-One/blob/env2pz/RouteRL/environment/environment.py) and is a [PettingZoo](https://pettingzoo.farama.org/index.html) AEC API environment.
- There are two types of agents in the environment and are both represented by the [BaseAgent](https://github.com/COeXISTENCE-PROJECT/Milestone-One/blob/env2pz/RouteRL/environment/agent.py) class.
  - Human drivers are simulated using human route-choice behavior from transportation research.
  - Automated vehicles (AVs) are the RL agents that aim to optimize their routes and learn the most efficient paths.
- It is compatible with popular RL libraries such as [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html) and [TorchRL](https://pytorch.org/rl/stable/tutorials/torchrl_demo.html).

For more details, check the documentation online.

<!-- end intro -->



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
