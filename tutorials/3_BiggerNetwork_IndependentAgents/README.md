# Large Network: Independent AV Agents

> In this tutorial, agents navigate a larger network. The default origin and destination points are specified in [`routerl/networks/default_ods.json`](../../routerl/networks/default_ods.json) and can be customized. We implement the automated vehicles' learning process with [TorchRL](https://github.com/pytorch/rl).

---

## Network Overview

> In these notebooks, we utilize the **Ingolstadt network** within our simulator, [SUMO](https://eclipse.dev/sumo/). Since agents exhibit **selfish behavior**, we employ **independent learning algorithms** to model their decision-making.

> Users can customize `TrafficEnvironment` by consulting [`routerl/environment/defaults.json`](../../routerl/environment/defaults.json), overriding the desired settings in a dictionary, and passing that dictionary as keyword arguments.

### Included Tutorials:

- **[IQL Tutorial.](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/tutorials/3_BiggerNetwork_IndependentAgents/iql_mutation.ipynb)** Uses **Independent Q-Learning (IQL)** ([IQL](https://web.media.mit.edu/~cynthiab/Readings/tan-MAS-reinfLearn.pdf)) as an initial baseline for training decentralized policies.

- **[IPPO Tutorial.](ippo_mutation.ipynb)** Implements **Independent Proximal Policy Optimization (IPPO)** ([IPPO](https://arxiv.org/pdf/2011.09533)), which has demonstrated strong benchmark performance in various tasks ([paper1](https://arxiv.org/abs/2103.01955), [paper2](https://arxiv.org/abs/2006.07869)).

- **[ISAC Tutorial.](isac_mutation.ipynb)** Uses **Independent SAC (ISAC)**, the multi-agent extension of **Soft Actor-Critic (SAC)** ([SAC](https://arxiv.org/abs/1801.01290)), which balances exploration and exploitation using entropy-regularized reinforcement learning.

---

### Ingolstadt Network Visualization
<p align="center">
  <img src="plots_saved/ingolstadt.png" alt="Ingolstadt network" width="700"/>
</p>
