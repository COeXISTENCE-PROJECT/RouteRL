"""
Observation functions for RL agents (AVs).
"""

from gymnasium.spaces import Box
import numpy as np
from abc import ABC, abstractmethod
import os
import pandas as pd
from typing import List, Dict, Any
import traci.constants as tc

from routerl.keychain import Keychain as kc
from .simulator import SumoSimulator


def pad_invalid_freeflows_for_observation(
    freeflows: Dict[tuple, list[float]],
    action_masks: Dict[tuple, np.ndarray] | None
) -> Dict[tuple, np.ndarray]:
    """
    Replace masked free-flow slots with a neutral value for observations.
    The invalid slots stay masked externally, so this only affects what the
    network sees. Using a per-OD max keeps the scale consistent and avoids zero-padding
    or giant placeholder values.
    """

    padded: Dict[tuple, np.ndarray] = {}
    for od, ff_values in freeflows.items():
        ff = np.asarray(ff_values, dtype=np.float32).copy()
        if action_masks is None:
            padded[od] = ff
            continue

        if od not in action_masks:
            raise ValueError(f"Missing action mask for OD {od}")

        mask = np.asarray(action_masks[od], dtype=bool).reshape(-1)
        if ff.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Free-flow length {ff.shape[0]} does not match mask length {mask.shape[0]} for OD {od}"
            )
        if not mask.any():
            raise ValueError(f"OD {od} has no valid routes in its action mask")

        valid_ff = ff[mask]
        pad_value = float(valid_ff.max())
        
        ff[~mask] = pad_value
        padded[od] = ff

    return padded


def get_ema(ff_time, values, max_length=10):
    """
    Compute the exponential moving average (EMA) given the list of
    observed travel times.
    """
    ema_val = ff_time

    values = values[-max_length:] if len(values) > max_length else values
    k = len(values)
    alpha = 2 / (k + 1)

    for tt in values:
        ema_val = (alpha * tt) + ((1 - alpha) * ema_val)

    return ema_val


class Observations(ABC):
    """Abstract base class for observation functions.

    Args:
        machine_agents_list (List[Any]): List of machine agents.
        human_agents_list (List[Any]): List of human agents.
    """

    def __init__(self, machine_agents_list: List[Any], human_agents_list: List[Any]) -> None:
        self.machine_agents_list = machine_agents_list
        self.human_agents_list = human_agents_list

    @abstractmethod
    def __call__(self, all_agents: List[Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def observation_space(self) -> Dict[str, Box]:
        """Define the observation space for the observation function.

        Returns:
            Dict[str, Box]: A dictionary where keys are agent IDs and values are Gym spaces.
        """

        pass


class PreviousAgentStart(Observations):
    """Observes the number of agents with the same origin-destination and start time within a threshold.

    Args:
        machine_agents_list (List[Any]): List of machine agents.
        human_agents_list (List[Any]): List of human agents.
        simulation_params (Dict[str, Any]): Simulation parameters.
        agent_params (Dict[str, Any]): Agent parameters.
    Attributes:
        observations (List[Any]): List of observations.
    """

    def __init__(
        self,
        machine_agents_list: List[Any],
        human_agents_list: List[Any],
        simulation_params: Dict[str, Any],
        agent_params: Dict[str, Any],
        ) -> None:

        super().__init__(machine_agents_list, human_agents_list)
        self.simulation_params = simulation_params
        self.agent_params = agent_params
        self.observations = self.reset_observation()

    def __call__(self, all_agents: List[Any]) -> Dict[str, Any]:
        for machine in self.machine_agents_list:
            observation = np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=int)

            for agent in all_agents:
                if (machine.id != agent.id and
                    machine.origin == agent.origin and
                    machine.destination == agent.destination and
                    abs(machine.start_time - agent.start_time) < machine.observed_span):
                    
                    observation[agent.last_action] += 1

            self.observations[str(machine.id)] = observation.tolist()

        return self.observations

    def reset_observation(self) -> Dict[str, np.ndarray]:
        """Reset observations to the initial state.

        Returns:
            Dict[str, np.ndarray]: A dictionary of initial observations for all machine agents.
        """

        return {
            str(agent.id): np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.float32)
            for agent in self.machine_agents_list
        }

    def observation_space(self) -> Dict[str, Box]:
        """Define the observation space for each machine agent.

        Returns:
            Dict[str, Box]: A dictionary where keys are agent IDs and values are Gym spaces.
        """

        return {
            str(agent.id): Box(
                low=0,
                high=len(self.human_agents_list) + len(self.machine_agents_list),
                shape=(self.simulation_params[kc.NUMBER_OF_PATHS],),
                dtype=np.float32
            )
            for agent in self.machine_agents_list
        }

    def agent_observations(self, agent_id: str, all_agents: List[Any]) -> np.ndarray:
        """Retrieve the observation for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observation array for the specified agent.
        """
        for machine in self.machine_agents_list:
            if machine.id == int(agent_id):
                break
            
        observation = np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)

        for agent in all_agents:
            if (machine.id != agent.id and
                machine.origin == agent.origin and
                machine.destination == agent.destination and
                machine.start_time > agent.start_time):
                
                observation[agent.last_action] += 1
        
        return observation


class PreviousAgentStartPlusStartTime(Observations):
    """Observes the number of agents with the same origin-destination and start time within a threshold
    and includes the start of the specific agent as well.
    """

    def __init__(
        self,
        machine_agents_list: List[Any],
        human_agents_list: List[Any],
        simulation_params: Dict[str, Any],
        agent_params: Dict[str, Any],
    ) -> None:
        """Initialize the observation function.

        Args:
            machine_agents_list (List[Any]): List of machine agents.
            human_agents_list (List[Any]): List of human agents.
            simulation_params (Dict[str, Any]): Dictionary of simulation parameters.
            agent_params (Dict[str, Any]): Dictionary of agent parameters.
        Returns:
            None
        """

        super().__init__(machine_agents_list, human_agents_list)
        self.simulation_params = simulation_params
        self.agent_params = agent_params
        self.observations = self.reset_observation()
        self.agent_vectors = {}

    def __call__(self, all_agents: List[Any]) -> Dict[str, Any]:
        """Generate observations for all agents.

        Args:
            all_agents (List[Any]): List of all agents.
        Returns:
            Dict[str, Any]: A dictionary of observations keyed by agent IDs.
        """

        """for machine in self.machine_agents_list:
            observation = np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)

            for agent in all_agents:
                if (machine.id != agent.id and
                    machine.origin == agent.origin and
                    machine.destination == agent.destination and
                    machine.start_time > agent.start_time):
                    
                    observation[agent.last_action] += 1

            self.observations[str(machine.id)] = np.concatenate(
                [
                    np.array([machine.start_time], dtype=np.int32),  # Start time as scalar
                    observation  # Vector of observations
                ]
            )"""

        return self.observations

    def reset_observation(self) -> Dict[str, np.ndarray]:
        """Reset observations to the initial state.

        Returns:
            obs (Dict[str, np.ndarray]): A dictionary of initial observations for all machine agents.
        """

        # Initialize agent vectors as zero arrays
        self.agent_vectors = {
            agent: np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)
            for agent in self.machine_agents_list
        }
        
        # Gather observations in a consistent format
        obs = {
            str(agent.id): np.concatenate(  # Combine start_time and vector into a single array
                [
                    np.array([agent.start_time], dtype=np.int32),  # Start time as scalar
                    self.agent_vectors[agent]  # Vector as array
                ]
            )
            for agent in self.machine_agents_list
        }

        self.observations = obs

        return obs

    def observation_space(self) -> Dict[str, Box]:
        """
        Define the observation space for each machine agent.

        Returns:
            Dict[str, Box]: A dictionary where keys are agent IDs and values are Gym spaces.
        """

        total_size = 1 + self.simulation_params[kc.NUMBER_OF_PATHS]

        return {
            str(agent.id): Box(
                low=0,
                high=np.inf,
                shape=(total_size,),  # Combined size for start_time and vector
                dtype=np.float32
            )
            for agent in self.machine_agents_list
        }
    
    def agent_observations(self, agent_id: str, all_agents: List[Any], agent_selection: str, travel_times: list) -> np.ndarray:
        """Retrieve the observation for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observation array for the specified agent.
        """
        for machine in self.machine_agents_list:
            if machine.id == int(agent_id):
                break

        # If the agent has already acted, return the observation that was previously calculated
        if agent_id != agent_selection:
            observation = self.observations[str(machine.id)]   

        # If the agent is about to act calculate its observation
        else:
            observation = np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)

            for agent in all_agents:
                if (machine.id != agent.id and
                    machine.origin == agent.origin and
                    machine.destination == agent.destination and
                    machine.start_time > agent.start_time):
                    
                    observation[agent.last_action] += 1

            observation = np.concatenate(([machine.start_time], observation))

            self.observations[str(machine.id)] = observation
        
        return observation
    

class PreviousAgentStartPlusStartTimeDetectorData(Observations):
    """Observes the number of agents with the same origin-destination and start time within a threshold
    and includes the start of the specific agent as well.
    """

    def __init__(
        self,
        machine_agents_list: List[Any],
        human_agents_list: List[Any],
        simulation_params: Dict[str, Any],
        plotter_params: Dict[str, Any],
        agent_params: Dict[str, Any],
        simulator: SumoSimulator
    ) -> None:
        """Initialize the observation function.

        Args:
            machine_agents_list (List[Any]): List of machine agents.
            human_agents_list (List[Any]): List of human agents.
            simulation_params (Dict[str, Any]): Dictionary of simulation parameters.
            agent_params (Dict[str, Any]): Dictionary of agent parameters.
        Returns:
            None
        """

        super().__init__(machine_agents_list, human_agents_list)
        self.simulation_params = simulation_params
        self.agent_params = agent_params
        self.plotter_params = plotter_params
        self.observations = self.reset_observation()
        self.agent_vectors = {}
        self.simulator = simulator

    def __call__(self, all_agents: List[Any]) -> Dict[str, Any]:
        """Generate observations for all agents.

        Args:
            all_agents (List[Any]): List of all agents.
        Returns:
            Dict[str, Any]: A dictionary of observations keyed by agent IDs.
        """

        return self.observations

    def reset_observation(self) -> Dict[str, np.ndarray]:
        """Reset observations to the initial state.

        Returns:
            obs (Dict[str, np.ndarray]): A dictionary of initial observations for all machine agents.
        """

        # Initialize agent vectors as zero arrays
        self.agent_vectors = {
            agent: np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)
            for agent in self.machine_agents_list
        }
        
        # Gather observations in a consistent format
        obs = {
            str(agent.id): np.concatenate(  # Combine start_time and vector into a single array
                [
                    np.array([agent.start_time], dtype=np.int32),  # Start time as scalar
                    self.agent_vectors[agent],  # Vector as array
                    np.array([0, 0], dtype=np.int32) # Detectors data is zero
                ]
            )
            for agent in self.machine_agents_list
        }

        self.observations = obs

        return obs

    def observation_space(self) -> Dict[str, Box]:
        """
        Define the observation space for each machine agent.

        Returns:
            Dict[str, Box]: A dictionary where keys are agent IDs and values are Gym spaces.
        """

        total_size = 1 + self.simulation_params[kc.NUMBER_OF_PATHS] + 2 # 2 detectors data are used here

        return {
            str(agent.id): Box(
                low=0,
                high=np.inf,
                shape=(total_size,),  # Combined size for start_time and vector
                dtype=np.float32
            )
            for agent in self.machine_agents_list
        }
    
    def agent_observations(self, agent_id: str, all_agents: List[Any], agent_selection: str, travel_times: list) -> np.ndarray:
        """Retrieve the observation for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observation array for the specified agent.
        """
        for machine in self.machine_agents_list:
            if machine.id == int(agent_id):
                break

        # If the agent hasn't steped yet return an "empty observation"
        # The agent hasn't acted yet so only the start time is meaningful
        if agent_id != agent_selection:

            observation = self.observations[str(machine.id)]

        # Calculate the observation 
        else:
            
            observation = np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)

            file_path = f"{self.plotter_params[kc.RECORDS_FOLDER] + '/' + kc.DETECTOR_STOPPED_VEHICLES}/stopped_vehicles{self.simulator.timestep}.csv"

            # If the file doesn't exist, fallback to previous timestep
            if not os.path.isfile(file_path):
                file_path = f"{self.plotter_params[kc.RECORDS_FOLDER] + '/' + kc.DETECTOR_STOPPED_VEHICLES}/stopped_vehicles{self.simulator.timestep - 1}.csv"

            df = pd.read_csv(file_path)

            # Filter and count vehicles per detector
            # In the TRY network I am interested on the detectors E1 and E7
            e7_count = df[df["detector"] == "E7_det"]["vehicle_id"].nunique()
            e1_count = df[df["detector"] == "E1_det"]["vehicle_id"].nunique()

            detector_data_array = np.array([e7_count, e1_count])

            # Calculate the decisions of the vehicles that have start time smaller than the start time of the specific agent.
            for agent in all_agents:
                if (machine.id != agent.id and
                    machine.origin == agent.origin and
                    machine.destination == agent.destination and
                    machine.start_time > agent.start_time):
                    
                    observation[agent.last_action] += 1

            observation = np.concatenate(([machine.start_time], observation, detector_data_array))

            self.observations[str(machine.id)] = observation

        return observation
    

class TripInfoWithETA(Observations):
    """Includes:
    - ETA forecast for each path option
    - Origin index
    - Destination index
    - Start time of the agent
    """

    def __init__(
        self,
        machine_agents_list: List[Any],
        human_agents_list: List[Any],
        simulation_params: Dict[str, Any],
        agent_params: Dict[str, Any],
        freeflows: Dict[tuple, float]
    ) -> None:
        """Initialize the observation function.

        Args:
            machine_agents_list (List[Any]): List of machine agents.
            human_agents_list (List[Any]): List of human agents.
            simulation_params (Dict[str, Any]): Dictionary of simulation parameters.
            agent_params (Dict[str, Any]): Dictionary of agent parameters.
        Returns:
            None
        """

        super().__init__(machine_agents_list, human_agents_list)
        self.NUM_PATHS = simulation_params[kc.NUMBER_OF_PATHS]
        self.OBS_SIZE = 3 + self.NUM_PATHS  # Start time + origin + destination + TT EMAs
        self.freeflows = freeflows
        self.observations = self.reset_observation()

    def __call__(self, all_agents: List[Any]) -> Dict[str, Any]:
        """Generate observations for all agents.

        Args:
            all_agents (List[Any]): List of all agents.
        Returns:
            Dict[str, Any]: A dictionary of observations keyed by agent IDs.
        """
        return self.observations

    def reset_observation(self) -> Dict[str, np.ndarray]:
        """Reset observations to the initial state.

        Returns:
            obs (Dict[str, np.ndarray]): A dictionary of initial observations for all machine agents.
        """
        obs = {
            str(agent.id): np.concatenate(
                [
                    np.array(self.freeflows[(agent.origin, agent.destination)], dtype=np.float32),  # Free flow time
                    np.array([int(agent.origin), int(agent.destination), int(agent.start_time)], dtype=np.float32)
                ]
            )
            for agent in self.machine_agents_list
        }
        self.observations = obs

        return obs

    def observation_space(self) -> Dict[str, Box]:
        """
        Define the observation space for each machine agent.

        Returns:
            Dict[str, Box]: A dictionary where keys are agent IDs and values are Gym spaces.
        """
        return {
            str(agent.id): Box(
                low=-1,
                high=np.inf,
                shape=(self.OBS_SIZE,),
                dtype=np.float32
            )
            for agent in self.machine_agents_list
        }

    def agent_observations(self, agent_id: str, all_agents: List[Any], agent_selection: str, travel_times: List[Any]) -> np.ndarray:
        """Retrieve the observation for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observation array for the specified agent.
        """
        machine = next((m for m in self.machine_agents_list if m.id == int(agent_id)), None)
        assert machine is not None, f"Observing machine with ID {agent_id} not found."

        observation = self.observations[str(machine.id)].copy()

        agent_dicts = list()
        for entry in travel_times:
            not_agent_itself = entry[kc.AGENT_ID] != machine.id
            same_origin = entry[kc.AGENT_ORIGIN] == machine.origin
            same_destination = entry[kc.AGENT_DESTINATION] == machine.destination
            earlier_departure = entry[kc.AGENT_START_TIME] <= machine.start_time
            if all((not_agent_itself, same_origin, same_destination, earlier_departure)):
                agent_dicts.append(entry.copy())

        # Sort by arrival time, later arrival more impact
        agent_dicts.sort(key=lambda x: (x[kc.AGENT_START_TIME]+ (x[kc.TRAVEL_TIME] * 60.0)))

        tt_lists = {idx: list() for idx in range(self.NUM_PATHS)}
        for entry in agent_dicts:
            path_idx = int(entry[kc.ACTION])
            travel_time = entry[kc.TRAVEL_TIME]
            tt_lists[path_idx].append(travel_time)
        for i in range(self.NUM_PATHS):
            ff_time = self.freeflows[(machine.origin, machine.destination)][i]
            previous_tts = tt_lists[i][:]
            observation[i] = get_ema(ff_time, previous_tts)

        # Every other entry in the obs is already set at reset
        observation = np.array(observation, dtype=np.float32) # Ensure dtype
        self.observations[str(machine.id)] = observation.copy()
        return observation

class TripInfoWithETAMaskNorm(Observations):
    """ETA observation with normalized values and optional action masks.

    This variant is intended for fixed-size action spaces where some route
    options may be unavailable for a given origin-destination pair.
    """

    def __init__(
        self,
        machine_agents_list: List[Any],
        human_agents_list: List[Any],
        simulation_params: Dict[str, Any],
        agent_params: Dict[str, Any],
        freeflows: Dict[tuple, float],
        action_masks=None,
        include_action_mask_in_obs=False,
    ) -> None:
        """Initialize the observation function.

        Args:
            machine_agents_list (List[Any]): List of machine agents.
            human_agents_list (List[Any]): List of human agents.
            simulation_params (Dict[str, Any]): Dictionary of simulation parameters.
            agent_params (Dict[str, Any]): Dictionary of agent parameters.
        Returns:
            None
        """

        Observations.__init__(self, machine_agents_list, human_agents_list)
        self.NUM_PATHS = simulation_params[kc.NUMBER_OF_PATHS]
        if action_masks is None:
            self.freeflows = freeflows
        else:
            self.freeflows = pad_invalid_freeflows_for_observation(freeflows, action_masks)

        self.action_masks = action_masks
        self.include_action_mask_in_obs = include_action_mask_in_obs
        self.eta_scale = self._compute_eta_scale(freeflows, action_masks)
        self.start_time_scale = max(float(simulation_params.get(kc.SIMULATION_TIMESTEPS, 1.0) or 1.0), 1.0)

        # Start time + origin + destination + TT EMAs + optional action masks
        self.BASE_OBS_SIZE = 3 + self.NUM_PATHS
        if include_action_mask_in_obs:
            self.BASE_OBS_SIZE += self.NUM_PATHS

        self.OBS_SIZE = self.BASE_OBS_SIZE

        self.observations = self.reset_observation()

    def __call__(self, all_agents: List[Any]) -> Dict[str, Any]:
        """Generate observations for all agents.

        Args:
            all_agents (List[Any]): List of all agents.
        Returns:
            Dict[str, Any]: A dictionary of observations keyed by agent IDs.
        """
        return self.observations

    def _get_mask_for_agent(self, agent):
        if self.action_masks is None:
            return np.ones(self.NUM_PATHS, dtype=np.float32) # all valid

        mask = self.action_masks.get((int(agent.origin), int(agent.destination)))

        if mask is None:
            raise RuntimeError(
                f"Missing action mask for agent {agent.id}: "
                f"OD=({agent.origin}, {agent.destination})"
            )

        return np.asarray(mask, dtype=np.float32)

    def reset_observation(self) -> Dict[str, np.ndarray]:
        """Reset observations to the initial state.

        Returns:
            obs (Dict[str, np.ndarray]): A dictionary of initial observations for all machine agents.
        """
    
        obs = {}

        for agent in self.machine_agents_list:
            eta = self._normalize_eta_values(self.freeflows[(int(agent.origin), int(agent.destination))])

            od_time = np.array(
                [int(agent.origin), int(agent.destination), self._normalize_start_time(agent.start_time)],
                dtype=np.float32,
            )

            # Action masks are initialized once in reset_observation and don't change throughout the experiment.
            if self.include_action_mask_in_obs:
                mask = self._get_mask_for_agent(agent)
                obs_vec = np.concatenate([eta, mask, od_time])
            else:
                obs_vec = np.concatenate([eta, od_time])

            obs[str(agent.id)] = obs_vec

        self.observations = obs
        return obs

    def observation_space(self) -> Dict[str, Box]:
        """
        Define the observation space for each machine agent.

        Returns:
            Dict[str, Box]: A dictionary where keys are agent IDs and values are Gym spaces.
        """
        return {
            str(agent.id): Box(
                low=-1,
                high=np.inf,
                shape=(self.OBS_SIZE,),
                dtype=np.float32
            )
            for agent in self.machine_agents_list
        }
    
    def agent_observations(self, agent_id: str, all_agents: List[Any], agent_selection: str, travel_times: List[Any]) -> np.ndarray:
        """Retrieve the observation for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observation array for the specified agent.
        """
        machine = next((m for m in self.machine_agents_list if m.id == int(agent_id)), None)
        assert machine is not None, f"Observing machine with ID {agent_id} not found."
            
        # observation = self.observations[str(machine.id)].copy()
        observation = self.observations[str(machine.id)][:self.BASE_OBS_SIZE].copy()
            
        agent_dicts = list()
        for entry in travel_times:
            not_agent_itself = entry[kc.AGENT_ID] != machine.id
            same_origin = entry[kc.AGENT_ORIGIN] == machine.origin
            same_destination = entry[kc.AGENT_DESTINATION] == machine.destination
            earlier_departure = entry[kc.AGENT_START_TIME] <= machine.start_time
            if all((not_agent_itself, same_origin, same_destination, earlier_departure)):
                agent_dicts.append(entry.copy())
                
        # Sort by arrival time, later arrival more impact
        agent_dicts.sort(key=lambda x: (x[kc.AGENT_START_TIME]+ (x[kc.TRAVEL_TIME] * 60.0)))
        
        tt_lists = {idx: list() for idx in range(self.NUM_PATHS)}
        for entry in agent_dicts:
            path_idx = int(entry[kc.ACTION])
            travel_time = entry[kc.TRAVEL_TIME]
            tt_lists[path_idx].append(travel_time)
        for i in range(self.NUM_PATHS):
            ff_time = self.freeflows[(int(machine.origin), int(machine.destination))][i]
            previous_tts = tt_lists[i][:]
            observation[i] = self._normalize_eta_value(get_ema(ff_time, previous_tts))

        # Action masks are initialized once in reset_observation and don't change throughout the experiment.
        # Past observations are copied and modified so the action masks persist.
         
        # Every other entry in the obs is already set at reset   
        observation = np.array(observation, dtype=np.float32) # Ensure dtype
        self.observations[str(machine.id)] = observation.copy()
        return observation

    def _compute_eta_scale(self, freeflows, action_masks) -> float:
        valid_values = []
        for od, ff_values in freeflows.items():
            ff = np.asarray(ff_values, dtype=np.float32).reshape(-1)
            if action_masks is not None:
                if od not in action_masks:
                    raise ValueError(f"Missing action mask for OD {od}")
                mask = np.asarray(action_masks[od], dtype=bool).reshape(-1)
                if ff.shape[0] != mask.shape[0]:
                    raise ValueError(
                        f"Free-flow length {ff.shape[0]} does not match mask length {mask.shape[0]} for OD {od}"
                    )
                values = ff[mask]
            else:
                values = ff[np.isfinite(ff) & (ff > 0.0) & (ff < 1e8)]

            if values.size:
                valid_values.append(values)

        if not valid_values:
            return 1.0

        scale = float(np.percentile(np.concatenate(valid_values), 95))
        return max(scale, 1.0)

    def _normalize_eta_value(self, value: float) -> float:
        return float(np.clip(float(value) / self.eta_scale, 0.0, 5.0))

    def _normalize_eta_values(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return np.clip(values / self.eta_scale, 0.0, 5.0).astype(np.float32, copy=False)

    def _normalize_start_time(self, value: float) -> float:
        return float(np.clip(float(value) / self.start_time_scale, 0.0, 5.0))
    
class TripInfoWithETARouteCongestion(TripInfoWithETAMaskNorm):
    """
    Extends TripInfoWithETA by appending per-route congestion summaries.
    That way, the policy does not need to discover route relevance from thousands of edge features,
    they are included in the per-action summaries directly.

    The class expects a `simulator` (SumoSimulator) instance so it can
    read `edge_ids`, `edge_subscription_vars` and `latest_edge_state`.
    """

    def __init__(
        self,
        machine_agents_list: List[Any],
        human_agents_list: List[Any],
        simulation_params: Dict[str, Any],
        agent_params: Dict[str, Any],
        freeflows: Dict[tuple, float],
        simulator: SumoSimulator,
        action_masks=None,
        include_action_mask_in_obs=False,
        include_eta=True,
    ) -> None:
        self.simulator = simulator
        self.paths_csv_path = self.simulator.paths_csv_file_path
        self.include_eta = bool(include_eta)

        # Number of features for each route (e.g. mean speed)
        self.route_feature_dim = 7
        self.edge_vec_len = 0

        super().__init__(machine_agents_list, human_agents_list, simulation_params, agent_params, freeflows, action_masks, include_action_mask_in_obs)

        if not self.include_eta:
            self.BASE_OBS_SIZE -= self.NUM_PATHS

        self.route_to_edges = self._load_route_edges()
        self.edge_vec_len = self.NUM_PATHS * self.route_feature_dim

        # Base observation + a summary for each route (each summary might consist of multiple features)
        self.OBS_SIZE = self.BASE_OBS_SIZE + self.edge_vec_len
        self.observations = self.reset_observation()

    def reset_observation(self) -> Dict[str, np.ndarray]:
        if self.include_eta:
            base = super().reset_observation()
        else:
            base = {
                str(agent.id): self._trip_observation_without_eta(agent)
                for agent in self.machine_agents_list
            }

        if self.edge_vec_len == 0:
            self.observations = base
            return base

        obs = {}
        # For each agent, appends a zero-filled vector of edge attributes (which is ok for now given the edge attributes)
        for k, v in base.items():
            edge_pad = np.zeros(self.edge_vec_len, dtype=np.float32)
            obs[k] = np.concatenate([np.array(v, dtype=np.float32), edge_pad])
        self.observations = obs
        return obs
    
    def agent_observations(self, agent_id: str, all_agents: List[Any], agent_selection: str, travel_times: List[Any]) -> np.ndarray:
        machine = next((m for m in self.machine_agents_list if m.id == int(agent_id)), None)
        assert machine is not None, f"Observing machine with ID {agent_id} not found."

        if self.include_eta:
            base_obs = super().agent_observations(agent_id, all_agents, agent_selection, travel_times)
        else:
            base_obs = self._trip_observation_without_eta(machine)

        if self.edge_vec_len == 0:
            return np.array(base_obs, dtype=np.float32)

        origin_id = int(machine.origin)
        destination_id = int(machine.destination)

        # Per-route congestion summaries using the global SUMO network per-edge state for a given agent (OD)
        values = []
        for action in range(self.NUM_PATHS):
            edges = self.route_to_edges.get((origin_id, destination_id, action), [])
            values.extend(self._summarize_route(edges)) # flat

        edge_vec = np.asarray(values, dtype=np.float32)
        if edge_vec.shape[0] != self.edge_vec_len:
            raise RuntimeError(
                f"Edge vector length mismatch: got {edge_vec.shape[0]}, expected {self.edge_vec_len}"
            )

        full_obs = np.concatenate([np.array(base_obs, dtype=np.float32), edge_vec])
        self.observations[str(agent_id)] = full_obs.copy()
        return full_obs

    def _trip_observation_without_eta(self, agent) -> np.ndarray:
        od_time = np.array(
            [int(agent.origin), int(agent.destination), self._normalize_start_time(agent.start_time)],
            dtype=np.float32,
        )
        if not self.include_action_mask_in_obs:
            return od_time
        return np.concatenate([self._get_mask_for_agent(agent), od_time])

    def _load_route_edges(self) -> Dict[tuple[int, int, int], list[str]]:
        if not self.paths_csv_path or not os.path.isfile(self.paths_csv_path):
            return {}

        paths = pd.read_csv(self.paths_csv_path)
        required_columns = {"origins", "destinations", "path"}
        if not required_columns.issubset(paths.columns):
            missing = sorted(required_columns - set(paths.columns))
            raise ValueError(f"Route congestion observation missing columns in {self.paths_csv_path}: {missing}")

        route_to_edges: Dict[tuple[int, int, int], list[str]] = {}
        if "cluster" in paths.columns:
            for row in paths.itertuples(index=False):
                origin = int(getattr(row, "origins"))
                destination = int(getattr(row, "destinations"))
                action = int(getattr(row, "cluster"))
                route_to_edges[(origin, destination, action)] = self._parse_path_edges(getattr(row, "path"))
            return route_to_edges

        action_indices = paths.groupby(["origins", "destinations"]).cumcount()
        for row, action in zip(paths.itertuples(index=False), action_indices):
            origin = int(getattr(row, "origins"))
            destination = int(getattr(row, "destinations"))
            route_to_edges[(origin, destination, int(action))] = self._parse_path_edges(getattr(row, "path"))

        return route_to_edges

    def _parse_path_edges(self, path_value) -> list[str]:
        if pd.isna(path_value):
            return []
        path = str(path_value).strip()
        if not path:
            return []
        separator = "," if "," in path else None
        return [edge.strip() for edge in path.split(separator) if edge.strip()]

    def _summarize_route(self, edges: list[str]) -> list[float]:
        if not edges:
            return [0.0] * self.route_feature_dim

        edge_state = getattr(self.simulator, "latest_edge_state", {}) or {}
        vehicle_counts = []
        halting_counts = []
        speeds = []
        occupancies = []

        for edge_id in edges:
            per_edge = edge_state.get(edge_id, {}) if isinstance(edge_state, dict) else {}
            vehicle_count = float(per_edge.get(tc.LAST_STEP_VEHICLE_NUMBER, 0.0) or 0.0)
            halting_count = float(per_edge.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0.0) or 0.0)
            speed = float(per_edge.get(tc.LAST_STEP_MEAN_SPEED, 0.0) or 0.0)
            occupancy = float(per_edge.get(tc.LAST_STEP_OCCUPANCY, 0.0) or 0.0)

            vehicle_counts.append(max(vehicle_count, 0.0))
            halting_counts.append(max(halting_count, 0.0))
            speeds.append(max(speed, 0.0))
            occupancies.append(max(occupancy, 0.0))

        vehicle_counts_arr = np.asarray(vehicle_counts, dtype=np.float32)
        halting_counts_arr = np.asarray(halting_counts, dtype=np.float32)
        speeds_arr = np.asarray(speeds, dtype=np.float32)
        occupancies_arr = np.asarray(occupancies, dtype=np.float32)

        vehicle_sum = float(vehicle_counts_arr.sum())
        halting_sum = float(halting_counts_arr.sum())
        if vehicle_sum > 0.0:
            mean_speed = float(np.average(speeds_arr, weights=vehicle_counts_arr))
        else:
            mean_speed = float(speeds_arr.mean()) if speeds_arr.size else 0.0

        active_edge_fraction = float((vehicle_counts_arr > 0.0).mean()) if vehicle_counts_arr.size else 0.0
        halted_fraction = halting_sum / max(vehicle_sum, 1.0)

        return [
            vehicle_sum,
            halting_sum,
            mean_speed,
            float(occupancies_arr.mean()) if occupancies_arr.size else 0.0,
            float(occupancies_arr.max()) if occupancies_arr.size else 0.0,
            active_edge_fraction,
            halted_fraction,
        ]

class RouteCongestion(TripInfoWithETARouteCongestion):
    """
    Like TripInfoWithETARouteCongestion but without the ETAs.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["include_eta"] = False
        super().__init__(*args, **kwargs)

class TripInfoWithETASumo(TripInfoWithETAMaskNorm):
    """
    Extends TripInfoWithETA by appending a flattened SUMO edge snapshot.

    The class expects a `simulator` (SumoSimulator) instance so it can
    read `edge_ids`, `edge_subscription_vars` and `latest_edge_state`.
    """

    def __init__(
        self,
        machine_agents_list: List[Any],
        human_agents_list: List[Any],
        simulation_params: Dict[str, Any],
        agent_params: Dict[str, Any],
        freeflows: Dict[tuple, float],
        simulator: SumoSimulator,
        action_masks=None,
        include_action_mask_in_obs=False
    ) -> None:
        self.simulator = simulator
        self.refresh_edge_metadata()

        super().__init__(machine_agents_list, human_agents_list, simulation_params, agent_params, freeflows, action_masks, include_action_mask_in_obs)

        self.OBS_SIZE = self.BASE_OBS_SIZE + self.edge_vec_len
        self.observations = self.reset_observation()

    def refresh_edge_metadata(self) -> None:
        # Full SUMO observations depend on the edge list discovered after SUMO
        # starts/subscribes, so the observation shape must be refreshed from the
        # simulator metadata before spaces/observations are rebuilt.
        self.edge_ids = list(getattr(self.simulator, "edge_ids", []))
        self.edge_subscription_vars = tuple(getattr(self.simulator, "edge_subscription_vars", ()))
        self.edge_vec_len = len(self.edge_ids) * len(self.edge_subscription_vars)

        if hasattr(self, "BASE_OBS_SIZE"):
            self.OBS_SIZE = self.BASE_OBS_SIZE + self.edge_vec_len

    def reset_observation(self) -> Dict[str, np.ndarray]:
        # Free flow times for each agent instead of empirically updated values
        base = super().reset_observation()
        if self.edge_vec_len == 0:
            return base

        obs = {}
        # For each agent, appends a zero-filled vector of edge attributes (which is ok for now given the edge attributes)
        for k, v in base.items():
            edge_pad = np.zeros(self.edge_vec_len, dtype=np.float32)
            obs[k] = np.concatenate([np.array(v, dtype=np.float32), edge_pad])
        self.observations = obs
        return obs

    def agent_observations(self, agent_id: str, all_agents: List[Any], agent_selection: str, travel_times: List[Any]) -> np.ndarray:
        # Base observation: ETA for each path option, origin, destination, start_time (for a given agent)
        base_obs = super().agent_observations(agent_id, all_agents, agent_selection, travel_times)
        if self.edge_vec_len == 0:
            return np.array(base_obs, dtype=np.float32)

        # Global SUMO network state observation (per-edge attributes)
        edge_state = getattr(self.simulator, "latest_edge_state", {}) or {}
        values = [] # edge attributes list, e.g. mean speed, number of vehicles etc.
        # Retrieve all attibutes dict for a given edge
        for edge_id in self.edge_ids:
            per_edge = edge_state.get(edge_id, {}) if isinstance(edge_state, dict) else {}
            # Append attributes one by one from the dict
            for var_id in self.edge_subscription_vars:
                raw_value = float(per_edge.get(var_id, 0.0) or 0.0)
                values.append(raw_value)

        # Flat vector of all attributes for all edges
        edge_vec = np.asarray(values, dtype=np.float32)
        if edge_vec.shape[0] != self.edge_vec_len:
            raise RuntimeError(
                f"Edge vector length mismatch: got {edge_vec.shape[0]}, expected {self.edge_vec_len}"
            )

        full_obs = np.concatenate([np.array(base_obs, dtype=np.float32), edge_vec])
        self.observations[str(agent_id)] = full_obs.copy()
        return full_obs
