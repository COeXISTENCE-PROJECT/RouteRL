# Large-Sized Network: Independent AV agents

> In this tutorial we use a big-scale netowk for agents navigation. The chosen origin and destination points are specified in this [file](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/routerl/networks/default_ods.json), and can be adjusted by users. In parallel, we define AV behaviors based on the agents' reward formulation and implement their learning process using the [TorchRL](https://github.com/pytorch/rl) library.

---

## Network Overview

> In these notebooks, we utilize the **Manhattan network** within our simulator, [SUMO](https://eclipse.dev/sumo/).  Since agents exhibit **selfish behavior**, we employ **independent learning algorithms** to model their decision-making.

### Included Tutorials:
- **[IPPO Tutorial](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/tutorials/3_BiggerNetwork_IndependentAgents/mappo_ippo_mutation.ipynb)**  
  Implements **Independent Proximal Policy Optimization (IPPO)** ([IPPO](https://arxiv.org/pdf/2011.09533)), which has demonstrated strong benchmark performance in various tasks ([paper1](https://arxiv.org/abs/2103.01955), [paper2](https://arxiv.org/abs/2006.07869)).


---

### Manhattan Network Visualization
<p align="center">
  <img src="../../docs/_static/manhattan.png" alt="Manhattan network" width="700"/>
</p>

---
