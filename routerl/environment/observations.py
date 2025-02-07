"""
Observation functions for RL agents (AVs).
"""

from gymnasium.spaces import Box
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

from routerl.keychain import Keychain as kc


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

    def agent_observations(self, agent_id: str) -> np.ndarray:
        """Retrieve the observation for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observation array for the specified agent.
        """

        return self.observations.get(agent_id, np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.float32))


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

        for machine in self.machine_agents_list:
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
            )

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
    
    def agent_observations(self, agent_id: str) -> np.ndarray:
        """Retrieve the observation for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observation array for the specified agent.
        """

        return self.observations.get(agent_id, np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32))