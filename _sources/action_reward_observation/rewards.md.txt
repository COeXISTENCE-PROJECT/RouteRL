---
title: Rewards
firstpage:
---

# Rewards

The rewards of the human agents is defined as the negative of their travel time. The reward of the autonomous vehicles (AVs) can vary depending on their specific behavior.

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