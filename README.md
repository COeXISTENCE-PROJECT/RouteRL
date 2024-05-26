# Overview

This project involves the development of a reinforcement learning environment where machine and human agents learn the optimal path to get from an origin to a destination.
The implementation includes several components, as depicted in the UML Class and Sequence Diagrams [HERE](https://miro.com/app/board/uXjVN4vGqSI=/?share_link_id=316593087566).

# How to run on servers?

See [here](server_scripts/how_to.md).

# Training setting

## Number of agents
- 1200 agents
- Humans: 823 | Machines: 377 
- Humans: Gawron | Machines: DQN (Single)
## Machines' objective
- **COMPETITIVE**: Minimize own TT, maximize humans'
- Also conducted alturistic, social, selfish and collaborative.
- See [results](results).
## Training episodes
- 6000 episodes, 4 phases
- Phase 1 (**Settle**) : Starts in episode 1
    - Humans: 1200
    - Only humans learn.
- Phase 2 (**Shock**) : Starts in episode 1000
    - Humans: 823  Machines: 377 
    - Only machines learn.
- Phase 3 (**Adapt**) : Starts in episode 4000
    - Humans: 823  Machines: 377
    - Both machines and humans learn.
- Phase 4 (**Exhibit**) : Starts in episode 5500
    - Humans: 823  Machines: 377
    - Noone learns.
## Training duration
- 16 hours, 18 minutes, 18 seconds 
- 9.78 seconds per episode in average
## Hardware
 - gpu=gpu:1
 - mem=64G
 - cpus-per-task=4
 - partition=dgx

# Results
#### *All plots smoothed by n=50*

## Travel times (in minutes)
![](readme_plots/travel_times.png)


## Distribution of Travel Times
![](readme_plots/tt_dist.png)


## Collected Mean Rewards
![](readme_plots/rewards.png)


## Mean Losses of DNNs of Machines 
#### (Throughout their learning)
![](readme_plots/losses.png)


## Simulation Timesteps
![](readme_plots/simulation_length.png)


## Picked Actions for OD Pairs
![](readme_plots/actions.png)


## Action Selection Shifts After Mutation
![](readme_plots/actions_shifts.png)


## Freeflow Times
![](readme_plots/ff_travel_time.png)