# Medium-sized network with the introduction of automated vehicles (AVs) behaviors and the application of collaborative MARL algorithms using TorchRL.

---

## Network

> In these notebooks, we utilize the Cologne network in our simulator [SUMO](https://eclipse.dev/sumo/), where agents-vehicles will determine their route choices.

<img src="../../docs/img/cologne.png" alt="Cologne network" width="700" />


---

## AV behaviors
As described in the [paper](https://openreview.net/pdf?id=88zP8xh5D2) , the reward function imposes a selected behavior on the agent. For an agent *k* with behavioral
parameters **φₖ ∈ ℝ⁴**, the reward is obtained as:

**rₖ = φₖ₁ · Tₖₒₛₙ + φₖ₂ · Tₖ₉ᵣₒᵤₚ + φₖ₃ · Tₖₒₜₕₑᵣ + φₖ₄ · Tₖₐₗₗ** , (3)

where **Tₖ** is a vector of travel time statistics provided to agent *k*, which contains:

- **Own travel time (Tₒₛₙ):** The amount of time the agent has spent in traffic.
- **Group’s travel time (T₉ᵣₒᵤₚ):** The average travel time of agents within the same group as  
  the given agent (e.g., AVs for an AV agent).
- **Other group’s travel time (Tₒₜₕₑᵣ):** The average travel time of agents within other groups  
  than the given agent (e.g., humans for an AV agent).
- **System-wide travel time (Tₐₗₗ):** The average travel time of all the agents in the traffic.


| **Behavior**     | ϕ₁  | ϕ₂  | ϕ₃  | ϕ₄  | **Interpretation**                        |
|-----------------|----|----|----|----|------------------------------------------------|
| Altruistic     | 0  | 0  | 0  | 1  | Minimize delay for everyone                  |
| Collaborative  | 0.5| 0.5| 0  | 0  | Minimize delay for oneself and one’s own group |
| Competitive    | 2  | 0  | -1 | 0  | Minimize self-delay and maximize for other group |
| Malicious      | 0  | 0  | -1 | 0  | Maximize delay for other group               |
| Selfish        | 1  | 0  | 0  | 0  | Minimize delay for oneself                   |
| Social        | 0.5| 0  | 0  | 0.5| Minimize delay for oneself and everyone      |


--- 
