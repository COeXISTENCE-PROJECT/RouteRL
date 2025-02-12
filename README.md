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

for episode in range(human_learning_episodes): # human learning 
    env.step()

env.mutation() # some human agents transition to AV agents

for agent in env.agent_iter(): # pettingzoo stepping loop
    observation, reward, termination, truncation, info = env.last()

    action = policy.sample() # we consider that we have a trained policy

    env.step(action)
 
env.plot_results() # plot the results
env.stop_simulation() # stop the connection with SUMO
```


## Documentation


* [Tutorials](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials):
  * [Quickstart](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials/1_Quickstart_TraffficEnvironment_Introduction).
  * [Medium network and AVs behaviors](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials/2_MediumNetwork_AVsBehaviors_TorchRL_CollaborativeAlgorithms).
  * [Big network and independent AV agents](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials/3_BiggerNetwork_IndependentAgents).
  * [Large-scale network](https://github.com/COeXISTENCE-PROJECT/RouteRL/tree/main/tutorials/4_VeryBigNetwork).


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
  author      = {Ahmet Onur Akman, Anastasia Psarou, Łukasz Gorczyca, 
  Zoltán Görgy Varga, Grzegorz Jamróz, Rafał Kucharski},
  title       = {RouteRL},
  year        = {2025},
  url         = {https://github.com/COeXISTENCE-PROJECT/RouteRL},
  publisher   = {GitHub},
  journal     = {GitHub repository}
}
```

