# Medium-Sized Network: AV Behaviors & Collaborative MARL with TorchRL

> In this tutorial we use a medium sized-network for agents navigation. The chosen origin and destination points are specified in this [file](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/routerl/networks/default_ods.json), and  can be adjusted by users. In parallel, we define AV behaviors based on the agents' reward formulation and implement their learning process using the [TorchRL](https://github.com/pytorch/rl) library.



---

## Network Overview

> In these notebooks, we utilize the **Cologne network** within our simulator, [SUMO](https://eclipse.dev/sumo/). As an initial baseline, we employ **malicious AV behavior**, where all AVs share the same reward, making it an ideal setup for algorithms designed to solve collaborative tasks.

> Users can customize parameters for the `TrafficEnvironment` class by consulting the [`routerl/environment/params.json`](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/4f4bc0a90d821e95b7193b00c93d6aaf10b34f41/routerl/environment/params.json) file. Based on its contents, they can create a dictionary with their preferred settings and pass it as an argument to the `TrafficEnvironment` class.

### Included Tutorials:

- **[VDN Tutorial.](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/tutorials/2_MediumNetwork_AVsBehaviors_TorchRL_CollaborativeAlgorithms/vdn_mutation.ipynb)** Uses **Value Decomposition Networks** ([VDN](https://arxiv.org/pdf/1706.05296)) for decentralized MARL training.

- **[QMIX Tutorial.](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/tutorials/2_MediumNetwork_AVsBehaviors_TorchRL_CollaborativeAlgorithms/qmix_mutation.ipynb)** Implements **QMIX** ([QMIX](http://arxiv.org/abs/1803.11485)), which leverages a mixing network with a monotonicity constraint.

- **[MAPPO Tutorial.](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/tutorials/2_MediumNetwork_AVsBehaviors_TorchRL_CollaborativeAlgorithms/mappo_mutation.ipynb)** Uses **Multi-Agent Proximal Policy Optimization** ([MAPPO](https://arxiv.org/abs/2103.01955)), the multi-agent adaptation of **PPO** ([PPO](https://arxiv.org/abs/1707.06347)).

---

### Cologne Network Visualization
<p align="center">
  <img src="../_static/cologne.png" alt="Cologne network" width="700"/>
</p>

---

## Defining Automated Vehicles Behavior Through Reward Formulations

As described in the **[paper](https://openreview.net/pdf?id=88zP8xh5D2)**, the reward function enforces a selected behavior on the agent. For an agent *k* with behavioral parameters **φₖ ∈ ℝ⁴**, the reward is defined as:

$$
r_k = \varphi_{k1} \cdot T_{\text{own}, k} + \varphi_{k2} \cdot T_{\text{group}, k} + \varphi_{k3} \cdot T_{\text{other}, k} + \varphi_{k4} \cdot T_{\text{all}, k}
$$


where **Tₖ** is a vector of travel time statistics provided to agent *k*, containing:

- **Own Travel Time** ($T_{\text{own}, k}$): The amount of time the agent has spent in traffic.
- **Group Travel Time** ($T_{\text{group}, k}$): The average travel time of agents in the same group (e.g., AVs for an AV agent).
- **Other Group Travel Time** ($T_{\text{other}, k}$): The average travel time of agents in other groups (e.g., humans for an AV agent).
- **System-wide Travel Time** ($T_{\text{all}, k}$): The average travel time of all agents in the traffic network.

---

## Behavioral Strategies & Objective Weightings

| **Behavior**    | **ϕ₁** | **ϕ₂** | **ϕ₃** | **ϕ₄** | **Interpretation**                                    |
|---------------|------|------|------|------|----------------------------------------------------|
| **Altruistic**     | 0    | 0    | 0    | 1    | Minimize delay for everyone                       |
| **Collaborative**  | 0.5  | 0.5  | 0    | 0    | Minimize delay for oneself and one’s own group    |
| **Competitive**    | 2    | 0    | -1   | 0    | Minimize self-delay & maximize delay for others  |
| **Malicious**      | 0    | 0    | -1   | 0    | Maximize delay for the other group               |
| **Selfish**        | 1    | 0    | 0    | 0    | Minimize delay for oneself                        |
| **Social**        | 0.5  | 0    | 0    | 0.5  | Minimize delay for oneself & everyone            |

