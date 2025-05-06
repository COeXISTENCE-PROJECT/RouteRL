# Large-Sized Network: Independent AV agents

> In this tutorial we use a big-scale netowk for agents navigation. The chosen origin and destination points are specified in this [file](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/routerl/networks/default_ods.json), and can be adjusted by users. In parallel, we define AV behaviors based on the agents' reward formulation and implement their learning process using the [TorchRL](https://github.com/pytorch/rl) library.

---

## Network Overview

> In these notebooks, we utilize the **Manhattan network** within our simulator, [SUMO](https://eclipse.dev/sumo/).  Since agents exhibit **selfish behavior**, we employ **independent learning algorithms** to model their decision-making.

> Users can customize parameters for the `TrafficEnvironment` class by consulting the [`routerl/environment/params.json`](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/4f4bc0a90d821e95b7193b00c93d6aaf10b34f41/routerl/environment/params.json) file. Based on its contents, they can create a dictionary with their preferred settings and pass it as an argument to the `TrafficEnvironment` class.

### Included Tutorials:
- **[IPPO Tutorial](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/tutorials/3_BiggerNetwork_IndependentAgents/mappo_ippo_mutation.ipynb)**  
  Implements **Independent Proximal Policy Optimization (IPPO)** ([IPPO](https://arxiv.org/pdf/2011.09533)), which has demonstrated strong benchmark performance in various tasks ([paper1](https://arxiv.org/abs/2103.01955), [paper2](https://arxiv.org/abs/2006.07869)).

---

### Manhattan Network Visualization
<p align="center">
  <img src="../_static/manhattan.png" alt="Manhattan network" width="700"/>
</p>

