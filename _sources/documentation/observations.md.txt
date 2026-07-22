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

### `TripInfoWithETA`

`TripInfoWithETA` represents each agent using one estimated travel time (ETA) per route, followed by the agent's origin index, destination index, and start time. The resulting vector has `NUMBER_OF_PATHS + 3` elements:

```text
[eta_0, ..., eta_(NUMBER_OF_PATHS-1), origin, destination, start_time]
```

At reset, the route ETAs are initialized from their free-flow travel times. When an agent observes, each ETA is updated with an exponential moving average of up to the ten most recent recorded travel times for the same origin-destination pair and route. Only other agents whose start time is no later than the observing agent's start time contribute to the estimate.

```{eval-rst}
.. autoclass:: routerl.environment.observations.TripInfoWithETA
    :members:
    :exclude-members: _private_method
    :no-index:
```

### `TripInfoWithETAMaskNorm`

`TripInfoWithETAMaskNorm` provides the same trip information with normalized ETA and start-time values. ETA values are divided by the 95th percentile of valid free-flow travel times, and the start time is divided by the configured number of simulation timesteps; both are clipped to the range `[0, 5]`. Origin and destination indices remain unnormalized.

For fixed-size action spaces, unavailable route slots are padded with the largest valid free-flow time for the corresponding origin-destination pair. If `include_action_mask_in_obs` is enabled, the route action mask is inserted between the ETA values and trip metadata:

```text
[normalized_eta_0, ..., normalized_eta_(N-1),
 action_mask_0, ..., action_mask_(N-1),
 origin, destination, normalized_start_time]
```

Without the action mask, the vector has `N + 3` elements; with the mask, it has `2N + 3` elements, where `N` is `NUMBER_OF_PATHS`.

```{eval-rst}
.. autoclass:: routerl.environment.observations.TripInfoWithETAMaskNorm
    :members:
    :exclude-members: _private_method
    :no-index:
```

### `TripInfoWithETARouteCongestion`

`TripInfoWithETARouteCongestion` extends `TripInfoWithETAMaskNorm` with seven current SUMO congestion features for each route:

1. total vehicle count,
2. total halting vehicle count,
3. mean speed, weighted by vehicle count when vehicles are present,
4. mean occupancy,
5. maximum occupancy,
6. fraction of route edges containing vehicles, and
7. fraction of vehicles that are halted.

The route-to-edge mapping is read from the simulator's paths CSV file. Routes without a usable edge mapping receive seven zeros. The vector has `BASE_OBS_SIZE + 7N` elements, where `BASE_OBS_SIZE` is the size of the normalized trip-information vector, including an action mask when configured.

```{eval-rst}
.. autoclass:: routerl.environment.observations.TripInfoWithETARouteCongestion
    :members:
    :exclude-members: _private_method
    :no-index:
```

### `RouteCongestion`

`RouteCongestion` uses the same per-route SUMO congestion summaries as `TripInfoWithETARouteCongestion`, but omits the ETA values. Its base trip information contains the optional action mask followed by the origin index, destination index, and normalized start time:

```text
[optional_action_mask, origin, destination, normalized_start_time,
 route_0_features, ..., route_(N-1)_features]
```

The vector has `3 + 7N` elements without an action mask and `3 + 8N` elements with one.

```{eval-rst}
.. autoclass:: routerl.environment.observations.RouteCongestion
    :members:
    :exclude-members: _private_method
    :no-index:
```

### `TripInfoWithETASumo`

`TripInfoWithETASumo` extends `TripInfoWithETAMaskNorm` with a flattened snapshot of the entire SUMO network. For each subscribed edge, it appends the latest vehicle count, mean speed, occupancy, and halting vehicle count, in the simulator's edge and subscription-variable order.

If there are `E` subscribed edges and `F` subscribed edge variables, the vector has `BASE_OBS_SIZE + E * F` elements. `F` is currently four. The edge metadata and observation shape are refreshed after SUMO starts and its edge subscriptions are available.

```{eval-rst}
.. autoclass:: routerl.environment.observations.TripInfoWithETASumo
    :members:
    :exclude-members: _private_method
    :no-index:
```
