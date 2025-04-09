"""
Observation functions for RL agents (AVs).
"""

from gymnasium.spaces import Box
import numpy as np
from abc import ABC, abstractmethod
import os
import pandas as pd
from typing import List, Dict, Any, Union

from routerl.keychain import Keychain as kc
from .simulator import SumoSimulator


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
    
    def agent_observations(self, agent_id: str, all_agents: List[Any], agent_selection: str) -> np.ndarray:
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
    
    def agent_observations(self, agent_id: str, all_agents: List[Any], agent_selection: str) -> np.ndarray:
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


class PreviousAgentStartPlusStartTimeMarginalCost(Observations):
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
    
    def compute_marginal_cost(self, agent_id, all_agents, travel_times_list):
        from .environment import TrafficEnvironment ## added here because there was circular import problem

        #print(all_agents, "\n\n\n", all_agents[0].start_time, "\n\n\n")
        print("I am agent ", agent_id, "\n\n\n")

        for machine in self.machine_agents_list:
            if machine.id == int(agent_id):
                break
        
        agents_to_calculate_marginal_cost = []

        for agent in all_agents:
            if agent.kind == kc.TYPE_MACHINE and agent.start_time < machine.start_time:
                agents_to_calculate_marginal_cost.append(agent)

        print("agents to calculate marginal cost", agents_to_calculate_marginal_cost, "\n\n\n")
        
        ## Delete each agent from the environment
        for agent in agents_to_calculate_marginal_cost:
            df = pd.read_csv(os.path.join(self.agent_params[kc.RECORDS_FOLDER], self.agent_params[kc.AGENTS_CSV_FILE_NAME]))

            df["id"] = df["id"].astype(int)
            df = df[df["id"] != int(agent.id)]

            df.to_csv(os.path.join(self.agent_params[kc.RECORDS_FOLDER], "agents2.csv"), index=False)

            actions = []
            for index, row in df.iterrows():
                for agent in all_agents:
                    if agent.start_time < machine.start_time and row['id'] == agent.id:
                        actions.append(agent.last_action)
                    
            print("actions are: ", actions, "\n\n\n")

            env_params = {
                "agent_parameters" : {
                    "new_machines_after_mutation": 10,
                    "agents_csv_file_name": "agents2.csv",

                    "human_parameters" :
                    {
                        "model" : "general_model",

                        "noise_weight_agent" : 0,
                        "noise_weight_path" : 0.8,
                        "noise_weight_day" : 0.2,

                        "beta" : -1,
                        "beta_k_i_variability" : 0.1,
                        "epsilon_i_variability" : 0.1,
                        "epsilon_k_i_variability" : 0.1,
                        "epsilon_k_i_t_variability" : 0.1,

                        "greedy" : 0.9,
                        "gamma_c" : 0.0,
                        "gamma_u" : 0.0,
                        "remember" : 1,

                        "alpha_zero" : 0.8,
                        "alphas" : [0.2]  
                    },
                    "machine_parameters" :
                    {
                        "behavior" : "cooperative",
                        "observation_type" : "previous_agents_plus_start_time_marginal_cost",
                    }
                },
                "simulator_parameters" : {
                    "network_name" : "two_route_yield",
                    "sumo_type" : "sumo",
                },  
                "plotter_parameters" : {
                    "smooth_by" : 50,
                    "phase_names" : [
                        "Human learning", 
                        "Mutation - Machine learning",
                        "Testing phase"
                    ]
                },
                "path_generation_parameters":
                {
                    "number_of_paths" : 4,
                    "beta" : -.5,
                    "visualize_paths" : True
                }
            }

            env = TrafficEnvironment(seed=42, create_agents=False, create_paths=True, marginal_cost=True, **env_params)
            env.start(use_subprocess=True)

            for agent, action in zip(all_agents, actions):
                agent.default_action = action

            env.step()

            print("\nafter one step\n", env.travel_times_list, "\n\n\n")
            print("after env.step\n")   

            env.stop_simulation() 
            print("Simulation is over\n\n")

        return 

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
    
    def agent_observations(self, agent_id: str, all_agents: List[Any], agent_selection: str, travel_times_list: List[Any]) -> np.ndarray:
        """Retrieve the observation for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observation array for the specified agent.
        """
        self.compute_marginal_cost(agent_id, all_agents, travel_times_list)

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
    