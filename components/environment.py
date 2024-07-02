from gymnasium.spaces import Box, Discrete
import functools
from copy import copy
import logging
import pandas as pd

from create_agents import create_agent_objects
from .simulator import SumoSimulator
from keychain import Keychain as kc
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
from .observations import PreviousAgentStart


logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class TrafficEnvironment(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "TrafficEnvironment",
    }
    """ A PettingZoo AECEnv interface for route planning using SUMO simulator.
    This environment is utilized for the training of human agents (rational decision-makers) and machine agents (reinforcement learning agents).
    See https://sumo.dlr.de/docs/ for details on SUMO.
    See https://pettingzoo.farama.org/ for details on PettingZoo. 
    Args:
        training_params (dict): Training parameters.
        environment_params (dict): Environment parameters.
        simulation_params (dict): Simulation parameters.
        agent_params (dict): Agent parameters.
        render_mode (str): The render mode.
    
    """
    def __init__(self,
                training_params,
                environment_params,
                simulation_params,
                agent_gen_params,
                agent_params, 
                render_mode=None):
        
        super().__init__()

        self.environment_params = environment_params
        self.agent_gen_params = agent_gen_params
        self.training_params = training_params
        self.simulation_params = simulation_params
        self.agent_params = agent_params
        self.render_mode = render_mode

        self.action_space_size = self.environment_params[kc.ACTION_SPACE_SIZE]

        self.simulator = SumoSimulator(simulation_params)
        logging.info("Simulator initiated!")

        self.agents = create_agent_objects(self.agent_params, self.get_free_flow_times())
        
        self.machine_agents = []
        self.human_agents = []
        self.possible_agents = []

        for agent in self.agents:
            if agent.kind == kc.TYPE_MACHINE:
                self.machine_agents.append(agent)
                self.possible_agents.append(agent.id)

            elif agent.kind == kc.TYPE_HUMAN:
                self.human_agents.append(agent)
            else:
                raise ValueError('[AGENT TYPE INVALID] Unrecognized agent type: ' + agent.kind)
            
        self.n_agents = len(self.possible_agents)
            
        if len(self.machine_agents) != 0:
            self._initialize_machine_agents(mutation=False)


        logging.info(f"There are {self.n_agents} machine agents in the environment.")
        logging.info(f"There are {len(self.human_agents)} human agents in the environment.")

        self.episode_actions = dict()


    def _initialize_machine_agents(self, mutation):
        """ Initialize the machine agents. """

        ## Sort machine agents based on their start_time
        sorted_machine_agents = sorted(self.machine_agents, key=lambda agent: agent.start_time)
        self.possible_agents = [str(agent.id) for agent in sorted_machine_agents]
        self.n_agents = len(self.possible_agents)

        print("self.possible_agents are: ", self.possible_agents)

        self.observation_obj = PreviousAgentStart(self.machine_agents, self.simulation_params, self.agent_params, self.training_params)

        self._observation_spaces = self.observation_obj.observation_space()

        self._action_spaces = {
            agent: Discrete(self.simulation_params[kc.NUMBER_OF_PATHS]) for agent in self.possible_agents
        }

        logging.info("\nMachine's observation space is: %s ", self._observation_spaces)
        logging.info("Machine's action space is: %s", self._action_spaces)

        

    #####################

    ##### CONTROL #####

    def start(self):
        self.simulator.start()

    def stop(self):
        self.simulator.stop()

    def reset(self):
        """Resets the environment."""
        self.episode_actions = dict()
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

        infos = {a: {}  for a in self.possible_agents}

        return self.observations, infos

    #####################

    ##### EPISODE OPS #####
    
    def get_observation(self):
        return self.simulator.timestep, self.episode_actions.values()


    def step(self, actions: list[tuple]):
        for agent, action in actions:
            action_dict = {kc.AGENT_ID: agent.id, kc.AGENT_KIND: agent.kind, kc.ACTION: action, \
                kc.AGENT_ORIGIN: agent.origin, kc.AGENT_DESTINATION: agent.destination, kc.AGENT_START_TIME: agent.start_time}
            self.simulator.add_vehice(action_dict)
            self.episode_actions[agent.id] = action_dict
        timestep, arrivals = self.simulator.step()
        travel_times = dict()
        for veh_id in arrivals:
            agent_id = int(veh_id)
            travel_times[agent_id] = {kc.TRAVEL_TIME : (timestep - self.episode_actions[agent_id][kc.AGENT_START_TIME]) / 60.0}
            travel_times[agent_id].update(self.episode_actions[agent_id])
        return travel_times.values()
    
    def close(self):
        """Close the environment and stop the SUMO simulation."""
        self.simulator.stop()

    def observe(self, agent):
        return self.observation_obj.agent_observations(agent)
    
    def render(self):
        pass
    
    #####################

    ##### DATA #####

    def get_free_flow_times(self):
        paths_df = pd.read_csv(self.simulator.paths_csv_path)
        origins = paths_df[kc.ORIGIN].unique()
        destinations = paths_df[kc.DESTINATION].unique()
        ff_dict = {(o, d): list() for o in origins for d in destinations}
        for _, row in paths_df.iterrows():
            ff_dict[(row[kc.ORIGIN], row[kc.DESTINATION])].append(row[kc.FREE_FLOW_TIME])
        return ff_dict


    #####################

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]