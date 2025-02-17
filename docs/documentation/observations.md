# Observation Functions

In this section, we describe how to implement and extend observation functions in `RouteRL`. Observation functions are responsible for defining the observations available to agents interacting with the environment. 

To create a new observation function, you must define a class that inherits from the `Observations` base class. Below, we outline the structure of the `Observations` class and provide details about its members and functionality:

---

## Observations Class

```{eval-rst}
.. autoclass:: routerl.environment.Observations
    :members:
    :exclude-members: _private_method
    :no-index:
```

### Default Observation Function: `PreviousAgentStart`

The `PreviousAgentStart` class serves as the default implementation for observation functions in `RouteRL`. This class is designed to monitor and manage the number of agents with identical origin-destination pairs and start times, operating within a predefined threshold. 

Details of the `PreviousAgentStart` class are provided below:

```{eval-rst}
.. autoclass:: routerl.environment.observations.PreviousAgentStart
    :members:  
    :exclude-members: _private_method
    :no-index:
```

### `PreviousAgentStartPlusStartTime`

The `PreviousAgentStartPlusStartTime` class is another observation functions in `RouteRL`. This class is designed to monitor and manage the number of agents with identical origin-destination pairs and start times, operating within a predefined threshold as well as the start time of the specific agent. 

Details of the `PreviousAgentStartPlusStartTime` class are provided below:

```{eval-rst}
.. autoclass:: routerl.environment.observations.PreviousAgentStartPlusStartTime
    :members:
    :exclude-members: _private_method
    :no-index:
```