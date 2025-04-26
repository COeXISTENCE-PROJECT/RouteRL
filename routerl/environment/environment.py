"""
PettingZoo environment for optimal route choice using SUMO simulator.

"""

import glob
import os

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from copy import deepcopy as dc
from gymnasium.spaces import Discrete

import functools
import logging
import numpy as np
import pandas as pd
import random

from routerl.environment import generate_agents
from routerl.environment import SumoSimulator
from routerl.environment import MachineAgent
from routerl.environment import PreviousAgentStart, PreviousAgentStartPlusStartTime, PreviousAgentStartPlusStartTimeDetectorData, Observations
from routerl.keychain import Keychain as kc
from routerl.services import plotter
from routerl.services import Recorder
from routerl.utilities import get_params

from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class TrafficEnvironment(AECEnv):
    """
    A PettingZoo AECEnv interface for optimal route choice using SUMO simulator. 
    This environment is designed for the training of human agents (rational decision-makers) 
    and machine agents (reinforcement learning agents).
    
    See `SUMO <https://sumo.dlr.de/docs/>`_ for details on SUMO. \n
    See `PettingZoo <https://pettingzoo.farama.org/>`_ for details on PettingZoo. 
    
    .. note::
        Users can configure the experiment with keyword arguments, see the structure below. 
        Moreover, users can provide custom demand data in ``training_records/agents.csv``.
        You can refer to the structure of such a file `here <https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/docs/_static/agents_example.csv>`_.

    Args:
        seed (int, optional): 
            Random seed for reproducibility. Defaults to ``23423``.
        create_agents (bool, optional):
            Whether to create agent data. Defaults to ``True``.
        create_paths (bool, optional):
            Whether to generate paths. Defaults to ``True``.
        **kwargs (dict, optional): 
            User-defined parameter overrides. These override default values 
            from ``defaults.json`` and allow experiment configuration.
            

    Keyword arguments (see the usage below):
    
        - agent_parameters (dict, optional):
            Agent settings.
            
            - num_agents (int, default=100):
                Total number of agents.
            
            - new_machines_after_mutation (int, default=25):
                Number of humans converted to machines.
            
            - machine_parameters (**dict**):
                Machine agent settings.
                
                - behavior (str, default="selfish"):
                    Route choice behavior.
                    Options: ``selfish``, ``social``, ``altruistic``, ``malicious``, ``competitive``, ``collaborative``.
                    
                - observed_span (int, default=300):
                    Time window considered for observations.
                    
                - observation_type (str, default="previous_agents_plus_start_time"):
                    Type of observation.
                    Options: ``previous_agents``, ``previous_agents_plus_start_time``.

            - human_parameters (**dict**): 
                Human agent settings.
                
                - model (str, default="gawron"):
                    Decision-making model (options: ``random``, ``gawron``, ``weighted``).
                    
                - noise_weight_agent (float, default=0.2):
                    Agent noise weight in the error term composition.
                    
                - noise_weight_path (float, default=0.6):
                    Path noise weight in the error term composition.
                    
                - noise_weight_day (float, default=0.2):
                    Day noise weight in the error term composition.
                    
                - beta (float, default=-1.0):
                    **Negative value**, multiplier of reward (travel time) used in utility, determines sensitivity.
                    
                - beta_k_i_variability (float, default=0):
                    Variance of normal distribution for which ``beta_k_i`` is drawn (computed in utility).
                    
                - epsilon_i_variability (float, default=0):
                    Variance of normal distribution from which error terms are drawn, first term.
                    
                - epsilon_k_i_variability (float, default=0):
                    Variance of normal distribution from which error terms are drawn, second term.
                    
                - epsilon_k_i_t_variability (float, default=0):
                    Variance of normal distribution from which error terms are drawn, third term. These three terms must sum to 1.
                    
                - greedy (float, default=1.0):
                    1 - exploration_rate, probability with which the choice are rational (argmax(U)) and not probabilistic (random).
                    
                - gamma_c (float, default=0):
                    Bounded rationality component. Expressed as relative increase in costs/utilities below which user do not change behaviour (do not notice it).
                    
                - gamma_u (float, default=0):
                    Bounded rationality component. Expressed as relative increase in costs/utilities below which user do not change behaviour (do not notice it).
                
                - remember (int, default=1):
                    Number of days remembered to learn from.
                    
                - alpha_zero (float, default=0.2):
                    Weight with which the recent experience is carried forward in learning.
                    
                - alphas (list[float], default=[0.8]):
                    Vector of weights for historically recorded reward in weighted average, **needs to be size of** ``remember``.

        - environment_parameters (dict, optional):
            Environment settings.
            
            - number_of_days (int, default=1):
                Number of days in the scenario.
                
            - save_every (int, default=1):
                Save the episode data to disk every X days.

        - simulator_parameters (dict, optional): 
            SUMO simulator settings.
            
            - network_name (str, default="csomor"):
                Network name (e.g., ``arterial``, ``cologne``, ``grid``)
                
            - custom_network_folder (str, default="NA"):
                In case of custom network, specify the folder name.
            
            - simulation_timesteps (int, default=180):
                Total simulation time in seconds.
            
            - sumo_type (str, default="sumo"):
                SUMO execution mode (``sumo`` or ``sumo-gui``).
                
            - stuck_time (int, default=600):
                Number of seconds to tolerate before `teleporting` a stopped vehicle to resolve gridlocks.

        - path_generation_parameters (dict, optional):
            Path generation settings.
            
            - number_of_paths (int, default=3):
                Number of routes per OD.
                
            - beta (float, default=-3.0):
                Sensitivity to travel time in path generation.
                
            - weight (str, default="time"):
                Optimization criterion.
                
            - num_samples (int, default=100):
                Number of samples for path generation.
                
            - origins (str | list[str], default="default"):
                Origin points from the network. (e.g., ``["-25166682#0", "-4936412"]``)
                
            - destinations (str | list[str], default="default"):
                Destination points from the network. (e.g., ``["-115604057#1", "-279952229#4"]``)
                
            - visualize_paths (bool, default=True):
                Whether to visualize generated paths. Visuals will be saved in the ``plotter_parameters/plots_folder``.

        - plotter_parameters (dict, optional): 
            Plotting & logging settings.
            
            - records_folder (str, default="training_records"):
                Directory for training records.
                
            - plots_folder (str, default="plots"):
                Directory for plots.
                
            - plot_choices (str, default="all"):
                Selection of plots to be generated. Options: ``none``, ``basic``, ``all``.
                
            - smooth_by (int, default=50): 
                Smoothing parameter for plots.
                
            - phases (list[int], default=[0, 100]):
                X-axis positions for phase markers.
                
            - phase_names (list[str], default=["Human learning", "Mutation - Machine learning"]):
                Phase names for labeling phase markers.
    
    Usage:
        
        .. rubric:: Case 1
        
        .. code-block:: text
        
            % Your file structure in the beginning
            project_directory/
            |-- your_script.py
            
        .. code-block:: python
        
            >>> # Environment initialization
            ... env = TrafficEnvironment(
            ...     seed=42,
            ...     agent_parameters={
            ...         "num_agents": 5, 
            ...         "new_machines_after_mutation": 1, 
            ...         "machine_parameters": {
            ...             "behavior": "selfish"
            ...             }},
            ...     simulator_parameters={"sumo_type": "sumo-gui"},
            ...     path_generation_parameters={"number_of_paths": 2}
            ... )
            
        .. code-block:: text
        
            % File structure after the initialization:
            project_directory/
            |-- your_script.py
            |-- training_records/
            |   |-- agents.csv
            |   |-- paths.csv
            |   |-- detector/
            |   |   |--             % to be populated during simulation
            |   |-- episodes/
            |   |   |--             % to be populated during simulation
            |-- plots/
            |   |-- 0_0.png 
            |   |-- ...             % visuals of generated paths for each OD
            |   |-- ...             % to be populated after the experiment
            
        .. raw:: html

            <hr style="border:1px solid #ccc; margin: 20px 0;">
        
        .. rubric:: Case 2
        
        .. code-block:: text
        
            % Your file structure in the beginning
            project_directory/
            |-- your_script.py
            |-- training_records/
            |   |-- agents.csv      % your custom demand, conforming to the structure
        
        .. warning::
            Demand data in ``agents.csv`` should be aligned with the specified 
            experiment settings (e.g., number of agents, number of origins and destinations, etc.).
            
        .. code-block:: python
        
            >>> env = TrafficEnvironment(
            ...     create_agents=False, # Environment will use your agent data
            ...     agent_parameters={
            ...         "new_machines_after_mutation": 10, 
            ...         "machine_parameters": {
            ...             "behavior": "selfish"
            ...             }},
            ...     simulator_parameters={"network_name": "arterial"},
            ...     path_generation_parameters={"number_of_paths": 3}
            ... )
            
        .. code-block:: text
        
            % File structure after the initialization:
            project_directory/
            |-- your_script.py
            |-- training_records/
            |   |-- agents.csv      % stays the same, used for agent generation
            |   |-- paths.csv
            |   |-- detector/
            |   |   |--             % to be populated during simulation
            |   |-- episodes/
            |   |   |--             % to be populated during simulation
            |-- plots/
            |   |-- 0_0.png 
            |   |-- ...             % visuals of generated paths for each OD
            |   |-- ...             % to be populated after the experiment
            
        .. warning::
            Same approach does not translate to path generation.\n
            ``paths.csv`` is mainly used for visualization purposes. 
            For SUMO to operate correctly, a ``route.rou.xml`` 
            should be generated inside the ``routerl/networks/<net_name>/`` folder.\n
            It is advised to generate paths in each experiment providing a random seed,
            or set ``create_paths=False`` only when above criteria is met.

    Attributes:
        day (int): Current day index in the simulation.
        human_learning (bool): Whether human agents are learning.
        number_of_days (int): Number of days to simulate.
        action_space_size (int): Size of the action space.
        recorder (Recorder): Object for recording simulation data.
        simulator (SumoSimulator): SUMO simulator instance.
        all_agents (list): List of all agent objects.
        machine_agents (list): List of all machine agent objects.
        human_agents (list): List of all human agent objects.
    """
    
    metadata = {
        "render_modes": ["human"],
        "name": "TrafficEnvironment",
    }

    def __init__(self,
                 seed: int = 23423,
                 create_agents: bool = True,
                 create_paths: bool = True,
                 save_detectors_info: bool = False,
                 **kwargs) -> None:

        super().__init__()
        self.render_mode = None

        # Read default parameters, update with kwargs
        defaults_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), kc.DEFAULTS_FILE)
        params = get_params(defaults_path, resolve=True, update=kwargs)

        self.environment_params = params[kc.ENVIRONMENT]
        self.simulation_params = params[kc.SIMULATOR]
        self.agent_params = params[kc.AGENTS]
        self.plotter_params = params[kc.PLOTTER]
        self.path_gen_params = params[kc.PATH_GEN] if create_paths else None

        self.travel_times_list = []
        self.day = 0
        self.human_learning = True
        self.machine_same_start_time = []
        self.actions_timestep = []
        self.save_detectors_info = save_detectors_info

        self.number_of_days = self.environment_params[kc.NUMBER_OF_DAYS]
        self.save_every = self.environment_params[kc.SAVE_EVERY]
        self.action_space_size = self.environment_params[kc.ACTION_SPACE_SIZE]
        self._set_seed(seed)

        self.recorder = Recorder(self.plotter_params)
        self.simulator = SumoSimulator(self.simulation_params, self.path_gen_params, seed, not create_agents, save_detectors_info)

        self.all_agents = generate_agents(self.agent_params, self.get_free_flow_times(), create_agents, seed)
        self.machine_agents = [agent for agent in self.all_agents if agent.kind == kc.TYPE_MACHINE]
        self.human_agents = [agent for agent in self.all_agents if agent.kind == kc.TYPE_HUMAN]
        self.possible_agents = list()

        if len(self.machine_agents):
            self._initialize_machine_agents()
        if not self.human_agents:
            self.human_learning = False
        logging.info(f"There are {len(self.human_agents)} human and {len(self.machine_agents)} machine agents.")

        self.episode_actions = dict()
        
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_futures = []

    def __str__(self):
        message = f"TrafficEnvironment with {len(self.all_agents)} agents.\
            \n{len(self.machine_agents)} machines and {len(self.human_agents)} humans.\
            \nMachines: {sorted(self.machine_agents, key=lambda agent: agent.id)}\
            \nHumans: {sorted(self.human_agents, key=lambda agent: agent.id)}"
        return message

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        logging.info(f"Seed set to {seed}.")

    def _initialize_machine_agents(self) -> None:

        ## Sort machine agents based on their start_time
        sorted_machine_agents = sorted(self.machine_agents, key=lambda agent: agent.start_time)
        self.possible_agents = [str(agent.id) for agent in sorted_machine_agents]
        self.n_agents = len(self.possible_agents)

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        ## Initialize the observation object
        self.observation_obj = self.get_observation_function()
        self._observation_spaces = self.observation_obj.observation_space()

        self._action_spaces = {
            agent: Discrete(self.simulation_params[kc.NUMBER_OF_PATHS]) for agent in self.possible_agents
        }

        logging.info("\nMachine's observation space is: %s ", self._observation_spaces)
        logging.info("Machine's action space is: %s", self._action_spaces)

    ################################
    ######## Control methods #######
    ################################

    def start(self) -> None:
        """Start the connection with SUMO.
        
        Returns:
            None
        """

        self.simulator.start()

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """Resets the environment.
        
        Args:
            seed (int, optional): Seed for random number generation. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.
            
        Returns:
            observations (dict): observations.
            infos (dict): dictionary of information for the agents.
        """
        self.episode_actions = dict()
        self.travel_times_list = list()
        self.actions_timestep = list()
        self.machine_same_start_time = list()
        self.simulator.reset()
        self.agents = copy(self.possible_agents)
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.rewards_humans = {agent.id: 0 for agent in self.human_agents}

        if len(self.machine_agents) > 0:
            self._agent_selector = agent_selector(self.possible_agents)
            self.agent_selection = self._agent_selector.next()
            self.observations = self.observation_obj.reset_observation()
        else:
            self.observations = {}

        infos = {a: {} for a in self.possible_agents}

        return self.observations, infos

    def step(self, machine_action: int = None) -> None:
        """Step method.

        Takes an action for the current agent (specified by `agent_selection`) and updates
        various parameters including rewards, cumulative rewards, terminations, truncations,
        infos, and agent_selection. Also updates any internal state used by `observe()`.

        Args:
            machine_action (int, optional):
                The action to be taken by the machine agent. Defaults to None.
            
        Returns:
            None
        """

        # If there are machines in the system
        if self.possible_agents:
            if (self.terminations[self.agent_selection]
                    or self.truncations[self.agent_selection]):
                # handles stepping an agent which is already dead
                # accepts a None action for the one agent, and moves the agent_selection to
                # the next dead agent,  or if there are no more dead agents, to the next live agent
                self._was_dead_step(machine_action)
                return

            agent = self.agent_selection

            # The cumulative reward of the last agent must be 0
            self._cumulative_rewards[agent] = 0
            self.simulation_loop(machine_action, agent)

            # Collect rewards if it is the last agent to act
            if self._agent_selector.is_last():
                # Increase day number
                self.day += 1

                # Calculate the rewards
                self._assign_rewards()

                # The episode ends when we complete episode_length days
                self.truncations = {agent: not (self.day % self.number_of_days) for agent in self.agents}
                self.terminations = {agent: not (self.day % self.number_of_days) for agent in self.agents}
                self.infos = {agent: {} for agent in self.agents}
                self.observations = self.observation_obj(self.all_agents)
                self._reset_episode()
            else:
                # no rewards are allocated until all players give an action
                self._clear_rewards()
                self.agent_selection = self._agent_selector.next()

            # Adds .rewards to ._cumulative_rewards
            self._accumulate_rewards()

        # If there are only humans in the system
        else:
            self.simulation_loop(machine_action=0, machine_id=0)
            self.day = self.day + 1
            self._assign_rewards()
            self._reset_episode()

    def close(self) -> None:
        """Not implemented.

        Returns:
            None
        """
        pass
    
    def stop_simulation(self) -> None:
        """End the simulation.

        Returns:
            None
        """

        self.simulator.stop()
        for future in self.pending_futures:
            future.result()
        self.executor.shutdown(wait=True)

    def observe(self, agent: str) -> np.ndarray:
        """Retrieve the observations for a specific agent.

        Args:
            agent (str): The identifier for the agent whose observations are to be retrieved.
            
        Returns:
            self.observation_obj.agent_observations(agent) (np.ndarray): The observations for the specified agent.
        """
        for machine in self.machine_agents:
            if str(machine.id) == agent:
                break

        # If the agent's turn hasn't come and the start time is bigger than the simulator timestep return an "empty observation"
        # The agent hasn't acted yet so only the start time is meaningful
        if agent != self.agent_selection and machine.start_time > self.simulator.timestep:
            array = np.zeros(self._observation_spaces[agent].shape[0] - 1)
            observation = np.concatenate((np.array([machine.start_time]), array))
            return observation
        
        return self.observation_obj.agent_observations(agent, self.all_agents, self.agent_selection)

    #########################
    ### Mutation function ###
    #########################

    def mutation(self, disable_human_learning: bool = True, mutation_start_percentile: int = 25) -> None:
        """Perform mutation by converting selected human agents into machine agents.

        This method identifies a subset of human agents that start after the 25th percentile of start times of
        other vehicles, removes a specified number of these agents, and replaces them with machine agents.

        Args:
            disable_human_learning (bool, default=True): Boolean flag to disable human agents.
            mutation_start_percentile (int, default=25): The percentile threshold for selecting human agents for mutation. Set to -1 to disable this filter.
            
        Returns:
            None
            
        Raises:
            ValueError: If there are insufficient human agents available for mutation.
        """

        logging.info("Mutation is about to happen!\n")
        logging.info("There were %s human agents.\n", len(self.human_agents))

        if mutation_start_percentile == -1:
            filtered_human_agents = self.human_agents.copy()
        else:
            start_times = [human.start_time for human in self.human_agents]
            percentile = np.percentile(start_times, mutation_start_percentile)
            filtered_human_agents = [human for human in self.human_agents if human.start_time > percentile]

        number_of_machines_to_be_added = self.agent_params[kc.NEW_MACHINES_AFTER_MUTATION]

        if len(filtered_human_agents) < number_of_machines_to_be_added:
            raise ValueError(
                f"Insufficient human agents for mutation. Required: {number_of_machines_to_be_added}, "
                f"Available: {len(filtered_human_agents)}.\n"
                f"Decrease the number of machines to be added after the mutation.\n"
            )

        for _ in range(0, number_of_machines_to_be_added):
            random_human = random.choice(filtered_human_agents)

            self.human_agents.remove(random_human)
            filtered_human_agents.remove(random_human)

            self.machine_agents.append(MachineAgent(random_human.id,
                                                    random_human.start_time,
                                                    random_human.origin,
                                                    random_human.destination,
                                                    self.agent_params[kc.MACHINE_PARAMETERS],
                                                    self.action_space_size))
            self.possible_agents.append(str(random_human.id))

        self.n_agents = len(self.possible_agents)
        self.all_agents = self.machine_agents + self.human_agents

        if disable_human_learning:  self.human_learning = False

        logging.info(f"Now there are {len(self.human_agents)} human agents.")

        self._initialize_machine_agents()

    #########################
    ##### Help functions ####
    #########################

    def get_observation(self) -> tuple:
        """Retrieve the current observation from the simulator.

        This method returns the current timestep of the simulation and the values of the episode actions.

        Returns:
            tuple: A tuple containing the current timestep and the episode actions.
        """

        return self.simulator.timestep, self.episode_actions.values()

    def _help_step(self, actions: list[tuple]) -> dict:

        for agent, action in actions:
            action_dict = {kc.AGENT_ID: agent.id,
                           kc.AGENT_KIND: agent.kind,
                           kc.ACTION: action,
                           kc.AGENT_ORIGIN: agent.origin,
                           kc.AGENT_DESTINATION: agent.destination,
                           kc.AGENT_START_TIME: agent.start_time}
            self.simulator.add_vehicle(action_dict)
            self.episode_actions[agent.id] = action_dict
        timestep, stopped_vehicles_info, arrivals, teleported = self.simulator.step()

        if self.save_detectors_info == True:
            self._save_detectors_info(stopped_vehicles_info)

        travel_times = dict()
        for veh_id in arrivals:
            if veh_id not in teleported:
                agent_id = int(veh_id)
                travel_times[agent_id] = ({kc.TRAVEL_TIME:
                                            (timestep - self.episode_actions[agent_id][kc.AGENT_START_TIME]) / 60.0})
                travel_times[agent_id].update(self.episode_actions[agent_id])
            
        for veh_id in teleported:
            agent_id = int(veh_id)
            travel_times[agent_id] = ({kc.TRAVEL_TIME: self.simulator.simulation_length / 60.0})
            travel_times[agent_id].update(self.episode_actions[agent_id])

        return travel_times.values()
    
    def _save_detectors_info(self, stopped_vehicles_info):
        folder = self.plotter_params[kc.RECORDS_FOLDER] + '/' + kc.DETECTOR_STOPPED_VEHICLES
        os.makedirs(folder, exist_ok=True)

        if (self.simulator.timestep == 1):
             [os.remove(f) for f in glob.glob(f"{folder}/*.csv")]

        csv_file_path = f"{folder}/stopped_vehicles{self.simulator.timestep - 1}.csv"

        df = pd.DataFrame(stopped_vehicles_info, columns=["time", "detector", "vehicle_id"])
        df.to_csv(csv_file_path, index=False)

    def _reset_episode(self) -> None:

        detectors_dict = self.simulator.reset()

        if self.possible_agents:
            self._agent_selector = agent_selector(self.possible_agents)
            self.agent_selection = self._agent_selector.next()

        if self.day % self.save_every == 0:
            dc_episode, dc_ep_observations, dc_agents, dc_detectors = dc(self.day), dc(self.travel_times_list), dc(self.all_agents), dc(detectors_dict)
            recording_task = self.executor.submit(self._record, dc_episode, dc_ep_observations, dc_agents, dc_detectors)
            self.pending_futures.append(recording_task)
        
        # Reset observations
        if len(self.machine_agents) > 0:
            self.observations = self.observation_obj.reset_observation()

        self.travel_times_list = []
        self.episode_actions = dict()

    def _assign_rewards(self) -> None:

        for agent in self.all_agents:
            if agent.kind == 'Human':
                reward = agent.get_reward(self.travel_times_list)
            else:
                reward = agent.get_reward(self.travel_times_list, group_vicinity=self.agent_params[kc.MACHINE_PARAMETERS][kc.GROUP_VICINITY])

            # Add the reward in the travel_times_list
            for agent_entry in self.travel_times_list:
                if agent.id == agent_entry[kc.AGENT_ID]:
                    self.travel_times_list.remove(agent_entry)
                    agent_entry[kc.REWARD] = reward
                    self.travel_times_list.append(agent_entry)

            # Save machine's rewards based on PettingZoo standards
            if agent.kind == 'AV':
                self.rewards[str(agent.id)] = reward

            # Human learning
            elif self.human_learning:
                agent.learn(agent.last_action, self.travel_times_list)

    ###########################
    ##### Simulation loop #####
    ###########################

    def simulation_loop(self, machine_action: int, machine_id: int) -> None:
        """This function contains the integration of the agent's actions to SUMO.

        We iterate through all the time steps of the simulation.
        For each timestep there are none, one or more than one agents type (humans, machines) that start.
        If more than one machine agents have the same start time, we break from this function because
        we need to take the agent's action from the STEP function.

        Args:
            machine_action (int): The id of the machine agent whose action is to be performed.
            machine_id (int): The id of the machine agent whose action is to be performed.
            
        Returns:
            None
        """

        agent_action = False
        while (
                self.simulator.timestep < self.simulation_params[kc.SIMULATION_TIMESTEPS]
                or len(self.travel_times_list) < len(self.all_agents)
        ):

            # If there are more than one machines with the same start time
            # the humans should act once
            if not self.actions_timestep:
                for human in self.human_agents:
                    if human.start_time == self.simulator.timestep:
                        action = human.act(0)
                        human.last_action = action
                        self.actions_timestep.append((human, action))

            for machine in self.machine_agents:
                if machine.start_time == self.simulator.timestep:

                    # In case there are machine agents that have the same start time, but it's not their turn
                    if str(machine.id) != machine_id:

                        # If some machines have the same start time, and they haven't acted yet
                        if (
                                (machine not in self.machine_same_start_time)
                                and not any(machine == item[0] for item in self.actions_timestep)
                        ):
                            self.machine_same_start_time.append(machine)
                        continue
                    else:
                        # Machine acting
                        machine.last_action = machine_action
                        self.actions_timestep.append((machine, machine_action))

                        # The machine acted should be deleted from the self.machine_same_start_time list
                        if machine in self.machine_same_start_time:
                            self.machine_same_start_time.remove(machine)

                        # If the machine isn't the last agent to act then we need to step again for the next agent
                        if not self._agent_selector.is_last():
                            agent_action = True

            # If all machines that have start time as the simulator timestep acted
            if not self.machine_same_start_time:
                travel_times = self._help_step(self.actions_timestep)

                for agent_dict in travel_times:
                    self.travel_times_list.append(agent_dict)

                self.actions_timestep = []
                self.machine_same_start_time = []

            # If the machine agent that had turn acted
            if agent_action:
                agent_action = False
                break

    ###########################
    ##### Free flow times #####
    ###########################

    def get_free_flow_times(self) -> dict:
        """Retrieve free flow times for all origin-destination pairs from the simulator paths data.

        Returns:
            ff_dict (dict): A dictionary where keys are tuples of origin and destination,
                            and values are lists of free flow times.
        """

        paths_df = pd.read_csv(self.simulator.paths_csv_file_path)
        origins = paths_df[kc.ORIGINS].unique()
        destinations = paths_df[kc.DESTINATIONS].unique()
        ff_dict = {(o, d): list() for o in origins for d in destinations}

        for _, row in paths_df.iterrows():
            ff_dict[(row[kc.ORIGINS], row[kc.DESTINATIONS])].append(row[kc.FREE_FLOW_TIME])

        return ff_dict

    ############################
    ##### Disc operations ######
    ############################

    def _record(self, episode: int, ep_observations: dict, agents: list, detectors_dict: dict) -> None:
        zero_space = [0] * self.action_space_size
        cost_tables = [
            {
                kc.AGENT_ID: agent.id,
                kc.COST_TABLE: getattr(agent.model, 'cost', zero_space) if hasattr(agent, 'model') else zero_space
            }
            for agent in agents
        ]
        self.recorder.record(episode, ep_observations, cost_tables, detectors_dict)

    def plot_results(self) -> None:
        """Method that plot the results of the simulation.

        Returns:
            None
        """

        plotter(self.plotter_params)

    ############################
    ### PettingZoo functions ###
    ############################

    def render(self) -> None:
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        """Method that returns the observation space of the agent.

        Args:
            agent (str): The agent name.
        Returns:
            self._observation_spaces[agent] (Any): The observation space of the agent.
        """

        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        """Method that returns the action space of the agent.

        Args:
            agent (str): The agent name.
        Returns:
            self._action_spaces[agent] (Any): The action space of the agent.
        """

        return self._action_spaces[agent]

    #####################################################
    ### Decide on the observation function to be used ###
    #####################################################

    def get_observation_function(self) -> Observations:
        """Returns an observation object based on the provided parameters.

        Returns:
            Observations: An observation object.
        Raises:
            ValueError: If model is unknown.
        """

        params = self.agent_params[kc.MACHINE_PARAMETERS]
        observation_type = params[kc.OBSERVATION_TYPE]
        if observation_type == kc.PREVIOUS_AGENTS_PLUS_START_TIME:
            return PreviousAgentStartPlusStartTime(self.machine_agents,
                                                   self.human_agents,
                                                   self.simulation_params,
                                                   self.agent_params)
        elif observation_type == kc.PREVIOUS_AGENTS:
            return PreviousAgentStart(self.machine_agents,
                                      self.human_agents,
                                      self.simulation_params,
                                      self.agent_params)
        elif observation_type == kc.PREVIOUS_AGENTS_PLUS_START_TIME_DETECTOR_DATA:
            if self.save_detectors_info == False:
                raise Exception("Detector info saving is disabled. Please set 'self.save_detectors_info = True' to proceed or change the observation type.")
            
            return PreviousAgentStartPlusStartTimeDetectorData(self.machine_agents,
                                      self.human_agents,
                                      self.simulation_params,
                                      self.plotter_params,
                                      self.agent_params,
                                      self.simulator)
        else:
            raise ValueError('[MODEL INVALID] Unrecognized observation type: ' + observation_type)
