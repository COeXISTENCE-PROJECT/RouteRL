# Milestone 1

## Overview

This project involves the development of a reinforcement learning environment where machine and human agents learn the optimal path to get from an origin to a destination.
The implementation includes several components, as depicted in the UML Class and Sequence Diagrams [HERE](https://miro.com/app/board/uXjVN4vGqSI=/?share_link_id=316593087566).

## How to run on servers?

See [here](server_scripts/how_to.md).

## Training results

### Number of agents
1000 agents ~ around 800 humans, 200 selfish machines, 100 malicious (disruptive?) machines
### Training episodes
8000 Episodes, first mutation at 1000, second at 4000
### Training duration
15 hours, 10 minutes, 23 seconds
### Specs
mem: 64G, cpus-per-task: 20, partition: cpu


## Collected Mean Rewards(in minutes):
![](readme_plots/rewards.png)


## Distribution of Rewards:
![](readme_plots/rewards_dist.png)


## Simulation Timesteps:
![](readme_plots/simulation_length.png)


## Picked Actions for OD Pairs
![](readme_plots/actions.png)


## Action Selection Shifts After Mutation
![](readme_plots/actions_shifts.png)


## Route Populations:
![](readme_plots/flows.png)


## Freeflow Times
![](readme_plots/ff_travel_time.png)