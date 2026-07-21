---
title: Observations
firstpage:
---

# Observations

RouteRL provides the observation classes described below. Users can also create custom observations by inheriting from the `Observations` base class.

## PreviousAgentStart
The observation for each agent is represented as a vector of size $\mathcal{A}_j$, as the action space for an agent $j$, corresponding to the number of available actions. The value at position $i$ in the vector indicates the number of agents that selected route $i$ in the current episode and have a start time earlier than the given agent.

$$
\mathbf{o} =
\begin{bmatrix}
o_0 \\
o_1 \\
\vdots \\
o_{\mathcal{A}_j-1}
\end{bmatrix}
$$

where each element $o_i$ is defined as:

$$
o_i = \sum_{\substack{j \in \mathcal{A}_i \\ t_j < t_{\text{agent}}}} 1
$$

- $\mathbf{o}$ is the observation vector of size $a$, where $a$ is the number of available actions (routes).
- $o_i$ represents the number of agents that chose route $i$ in the current episode and have a start time earlier than the given agent.
- The summation iterates over all agents $j$ who selected route $i$ (denoted as $j \in \mathcal{A}_i$).
- The condition $t_j < t_{\text{agent}}$ ensures we only count agents that started earlier than the given agent.


## PreviousAgentStartPlusStartTime

This observation has the similar structure as the `PreviousAgentStart` but additionally includes the agent's own start time in the first position of the vector.

$$
\mathbf{o} =
\begin{bmatrix}
t_{\text{agent}} \\
o_0 \\
o_1 \\
\vdots \\
o_{\mathcal{A}_j-1}
\end{bmatrix}
$$

where each element $o_i$ is defined as:

$$
o_i = \sum_{\substack{j \in \mathcal{A}_i \\ t_j < t_{\text{agent}}}} 1
$$

- $\mathbf{o}$ is the observation vector of size $\mathcal{A}_j + 1$.
- The **first element** $t_{\text{agent}}$ represents the start time of the given agent.
- Each subsequent element $o_i$ represents the number of agents that selected route $i$ in the current episode and have a start time earlier than $t_{\text{agent}}$.
- The summation iterates over all agents $j$ who chose route $i$ (denoted as $j \in \mathcal{A}_i$), ensuring that only those with **earlier start times** are counted.


## TripInfoWithETA

`TripInfoWithETA` combines route-level estimated travel times with information about the observing trip. For $N$ configured routes, the observation is

$$
\mathbf{o} =
\begin{bmatrix}
\widehat{T}_0 \\
\vdots \\
\widehat{T}_{N-1} \\
\text{origin} \\
\text{destination} \\
t_{\text{agent}}
\end{bmatrix}.
$$

The observation therefore has $N + 3$ elements. Initially, each $\widehat{T}_i$ is the free-flow travel time of route $i$. It is subsequently updated using an exponential moving average (EMA) of up to the ten most recent recorded travel times from other agents that:

- have the same origin and destination,
- selected route $i$, and
- started no later than the observing agent.

For $k$ recorded travel times, the EMA uses $\alpha = 2/(k+1)$ and starts from the route's free-flow travel time. If there are no matching records, the estimate remains at the free-flow value.


## TripInfoWithETAMaskNorm

`TripInfoWithETAMaskNorm` normalizes the ETA and start-time values from `TripInfoWithETA` and supports fixed-size action spaces in which some route slots may be unavailable for a particular origin-destination pair.

ETA values are divided by the 95th percentile of all valid free-flow travel times and clipped to $[0, 5]$. The start time is divided by the configured number of simulation timesteps and clipped to the same range. Origin and destination indices are not normalized. Invalid route slots are assigned the largest valid free-flow time for the corresponding origin-destination pair; the action mask still determines which routes may be selected.

Without an action mask in the observation, the layout is

$$
\mathbf{o} =
\begin{bmatrix}
\widetilde{T}_0 \\
\vdots \\
\widetilde{T}_{N-1} \\
\text{origin} \\
\text{destination} \\
\widetilde{t}_{\text{agent}}
\end{bmatrix},
$$

which has $N + 3$ elements. When `include_action_mask_in_obs` is enabled, the mask is placed after the ETA values:

$$
\mathbf{o} =
\begin{bmatrix}
\widetilde{T}_0 \\
\vdots \\
\widetilde{T}_{N-1} \\
m_0 \\
\vdots \\
m_{N-1} \\
\text{origin} \\
\text{destination} \\
\widetilde{t}_{\text{agent}}
\end{bmatrix},
$$

where $m_i$ is one for an available route and zero for an unavailable route. This form has $2N + 3$ elements.


## TripInfoWithETARouteCongestion

`TripInfoWithETARouteCongestion` appends a seven-element congestion summary for every route to the `TripInfoWithETAMaskNorm` observation. For route $i$, the summary is

$$
\mathbf{c}_i =
\begin{bmatrix}
\text{vehicle count} \\
\text{halting vehicle count} \\
\text{mean speed} \\
\text{mean occupancy} \\
\text{maximum occupancy} \\
\text{active-edge fraction} \\
\text{halted-vehicle fraction}
\end{bmatrix}.
$$

The mean speed is weighted by the number of vehicles on each edge when vehicles are present. The active-edge fraction is the proportion of route edges containing at least one vehicle, and the halted-vehicle fraction is the total number of halted vehicles divided by the total vehicle count. Mean and maximum occupancy are calculated across the route's edges.

The complete observation is

$$
\mathbf{o}_{\text{route}} =
\begin{bmatrix}
\mathbf{o}_{\text{trip}} \\
\mathbf{c}_0 \\
\vdots \\
\mathbf{c}_{N-1}
\end{bmatrix},
$$

where $\mathbf{o}_{\text{trip}}$ is the normalized trip-information observation, optionally including its action mask. The resulting size is `BASE_OBS_SIZE` $+\,7N$. Route edges are obtained from the paths CSV file; if no usable mapping exists for a route, its seven congestion values are zero.


## RouteCongestion

`RouteCongestion` provides the same per-route congestion summaries as `TripInfoWithETARouteCongestion`, but leaves out the ETA values. Without an action mask, its layout is

$$
\mathbf{o} =
\begin{bmatrix}
\text{origin} \\
\text{destination} \\
\widetilde{t}_{\text{agent}} \\
\mathbf{c}_0 \\
\vdots \\
\mathbf{c}_{N-1}
\end{bmatrix},
$$

with $3 + 7N$ elements. If action masks are included, the $N$ mask values are placed before the origin, destination, and normalized start time, producing $3 + 8N$ elements.


## TripInfoWithETASumo

`TripInfoWithETASumo` combines `TripInfoWithETAMaskNorm` with a flattened snapshot of the entire SUMO network. For every subscribed edge, the current implementation appends these values in order:

1. vehicle count,
2. mean speed,
3. occupancy, and
4. halting vehicle count.

If the simulator subscribes to $E$ edges and $F$ edge variables, the complete observation size is `BASE_OBS_SIZE` $+\,EF$. Currently, $F=4$. Edge features are flattened in the order of the simulator's edge list and, within each edge, the subscription-variable order shown above. Before SUMO provides an edge snapshot, these appended values are initialized to zero. The edge metadata and observation shape are refreshed after SUMO starts.
