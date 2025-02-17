---
title: Observations
firstpage:
---

# Observations

RouteRL contains two observation classes `PreviousAgentStart` and `PreviousAgentStartPlusStartTime`. The users can create their own observation by using the class `Observations`.

## PreviousAgentStart
The observation for each agent is represented as a vector of size $a$
a, corresponding to the number of available actions. The value at position $i$ in the vector indicates the number of agents that selected route $i$ in the current episode and have a start time earlier than the given agent.

$$
\mathbf{o} =
\begin{bmatrix}
o_0 \\
o_1 \\
\vdots \\
o_{a-1}
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
