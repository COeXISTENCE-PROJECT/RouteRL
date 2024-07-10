from gymnasium.spaces import Box
import numpy as np

from abc import ABC, abstractmethod
from keychain import Keychain as kc


class Observations(ABC):
    """Abstract base class for observation functions."""

    def __init__(self, machine_agents_list, human_agents_list):
        """Initialize observation function."""
        self.machine_agents_list = machine_agents_list
        self.human_agents_list = human_agents_list

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class PreviousAgentStart(Observations):
    """Observation function that returns the start time of the previous agent."""

    def __init__(self, machine_agents_list, human_agents_list, simulation_params, agent_params, training_params):
        """Initialize observation function."""
        super().__init__(machine_agents_list, human_agents_list)

        self.simulation_params = simulation_params
        self.agent_params = agent_params
        self.training_params = training_params

    def __call__(self, all_agents):
        """Return the observations of all agents."""

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
    
    def reset_observation(self):
        """Reset the observations."""
        self.observations = {
            str(a.id): np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.float32) for a in self.machine_agents_list
        }
        return self.observations

    def observation_space(self):
        """Return the observation space of the observation function."""
        observations = {
            str(agent.id): Box(low=0, high=len(self.human_agents_list) + len(self.machine_agents_list), shape=(3,), dtype=np.float32) for agent in self.machine_agents_list 
        }
        return observations
    
    def agent_observations(self, agent):
        """Return the observations of a specific agent."""
        return self.observations[agent] 