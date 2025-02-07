# Introduction to our multi-agent PettingZoo framework and its functionalities

---

In this notebook we introduce the [TrafficEnvironment](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/routerl/environment/environment.py) class, a [PettingZoo](https://pettingzoo.farama.org/index.html) AEC API environment.

The [TrafficEnvironment](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/routerl/environment/environment.py) class communicates with [SUMO](https://eclipse.dev/sumo/), an open-source realistic traffic simulator.

<img src="../../docs/img/two_route_net_1.png" alt="two-route-network" />

<img src="../../docs/img/two_route_net_1_2.png" alt="two-route-network" />


---

In these notebooks, we use a two-route network in our simulator [SUMO](https://eclipse.dev/sumo/), where agents (vehicles) navigate from their predefined origin to their predefined destination point, aiming to determine the fastest route.

<img src="../../docs/img/two_route_yield.png" alt="two-route-network" />


## Related work

> Some methods have utilized MARL for optimal route choice (Thomasini et al. [2023](https://alaworkshop2023.github.io/papers/ALA2023_paper_69.pdf/)). These approaches
are typically based on macroscopic traffic simulations, which model relationships among traffic
flow characteristics such as density, flow, and mean speed of a traffic stream. In contrast, our
problem employs a microscopic model, which focuses on interactions between individual vehicles.

> Additionally, a method proposed by (Tavares and Bazzan [2012](https://www.researchgate.net/publication/235219033_Reinforcement_learning_for_route_choice_in_an_abstract_traffic_scenario)) addresses optimal route choice at the microscopic level, where rewards are generated through a predefined function. In contrast, in our approach, rewards are provided dynamically by a continuous traffic simulator.