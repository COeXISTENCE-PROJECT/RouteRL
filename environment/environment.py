from gymnasium.spaces import Discrete
import functools
from copy import copy
from copy import deepcopy as dc
import logging
import pandas as pd
import threading

from create_agents import create_agent_objects
from .simulator import SumoSimulator
from keychain import Keychain as kc
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
from utilities import show_progress_bar
from .observations import PreviousAgentStart


from services.recorder import Recorder


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
                phases_params,
                render_mode=None):
        
        super().__init__()

        self.environment_params = environment_params
        self.agent_gen_params = agent_gen_params
        self.training_params = training_params
        self.simulation_params = simulation_params
        self.agent_params = agent_params
        self.render_mode = render_mode
        self.phases_params = phases_params
        self.travel_times_df = pd.DataFrame(columns=['id', 'travel_time'])
        self.travel_times_dict = dict()
        self.travel_times_list = []
        self.action_cols = [kc.AGENT_ID, kc.AGENT_KIND, kc.ACTION, kc.AGENT_ORIGIN, kc.AGENT_DESTINATION, kc.AGENT_START_TIME]
        self.day = 0
        self.human_learning = True
        self.machine_same_start_time = []
        self.actions_timestep = []

        """ runner attributes """
        self.num_episodes = self.training_params[kc.NUM_EPISODES]
        self.phases = self.training_params[kc.PHASES]
        self.phase_names = self.training_params[kc.PHASE_NAMES]
        self.frequent_progressbar = self.training_params[kc.FREQUENT_PROGRESSBAR_UPDATE]
        self.remember_every = self.training_params[kc.REMEMBER_EVERY]

        """ phase """
        #self.phase_management = HumanLearning_Mutation(self.phases_params)
        
        """ recorder attributes """
        self.remember_episodes = [ep for ep in range(self.remember_every, self.num_episodes+1, self.remember_every)]
        self.remember_episodes += [1, self.num_episodes] + [ep-1 for ep in self.phases] + [ep for ep in self.phases]
        self.remember_episodes = set(self.remember_episodes)
        self.recorder = Recorder()
        
        self.curr_phase = -1

        #############################

        self.action_space_size = self.environment_params[kc.ACTION_SPACE_SIZE]

        self.simulator = SumoSimulator(simulation_params)
        logging.info("Simulator initiated!")

        self.all_agents = create_agent_objects(self.agent_params, self.get_free_flow_times())
        
        self.machine_agents = []
        self.human_agents = []
        self.possible_agents = []

        for agent in self.all_agents:
            if agent.kind == kc.TYPE_MACHINE:
                self.machine_agents.append(agent)

            elif agent.kind == kc.TYPE_HUMAN:
                self.human_agents.append(agent)
            else:
                raise ValueError('[AGENT TYPE INVALID] Unrecognized agent type: ' + agent.kind)
            
        if len(self.machine_agents) != 0:
            self._initialize_machine_agents()


        logging.info(f"There are {len(self.machine_agents)} machine agents in the environment.")
        logging.info(f"There are {len(self.human_agents)} human agents in the environment.")

        self.episode_actions = dict()


    def _initialize_machine_agents(self):
        """ Initialize the machine agents. """

        same_travel_time = 0
        for agent in self.machine_agents:
            for agent2 in self.machine_agents:
                if(agent.start_time == agent2.start_time) and (agent.id != agent2.id):
                    same_travel_time += 1
        
        ## Sort machine agents based on their start_time
        sorted_machine_agents = sorted(self.machine_agents, key=lambda agent: agent.start_time)
        self.possible_agents = [str(agent.id) for agent in sorted_machine_agents]
        self.n_agents = len(self.possible_agents)

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.observation_obj = PreviousAgentStart(self.machine_agents, self.human_agents, self.simulation_params, self.agent_params, self.training_params)

        self._observation_spaces = self.observation_obj.observation_space()

        self._action_spaces = {
            agent: Discrete(self.simulation_params[kc.NUMBER_OF_PATHS]) for agent in self.possible_agents
        }

        logging.info("\nMachine's observation space is: %s ", self._observation_spaces)
        logging.info("Machine's action space is: %s", self._action_spaces)

    
    #############################

    ##### Simulator control #####

    def start(self):
        self.simulator.start()

    def stop(self):
        self.simulator.stop()


    ################################

    ##### PettingZoo functions #####

    def reset(self, seed=None, options=None):
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

    
    def step(self, machine_action):
        """
        This function takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
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
            self.day= self.day+ 1

            # Calculate the rewards
            self._assign_rewards() 

            # The episode ends when we complete episode_length days
            self.truncations = {
                agent: self.day% self.training_params[kc.EPISODE_LENGTH] for agent in self.agents
            }

            self.terminations = {
                agent: self.day% self.training_params[kc.EPISODE_LENGTH] for agent in self.agents
            }

            self.info = {
                agent: {} for agent in self.agents
            }

            self.observations = self.observation_obj(self.all_agents)
            self._reset_episode()

        else:
            # no rewards are allocated until all players give an action
            self._clear_rewards()

            self.agent_selection = self._agent_selector.next()

        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


    def close(self):
        """Close the environment and stop the SUMO simulation."""
        self.simulator.stop()

    def observe(self, agent):
        return self.observation_obj.agent_observations(agent)
    
    def render(self):
        pass

    #########################

    ##### Help functions #####
    
    def get_observation(self):
        return self.simulator.timestep, self.episode_actions.values()


    def help_step(self, actions: list[tuple]):
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
    

    def _reset_episode(self):
        self.simulator.reset()

        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self._agent_selector.next()

        phase_start_time = 0

        recording_task = threading.Thread(target=self._record, args=(self.day, self.travel_times_list, phase_start_time, self.all_agents))
        recording_task.start()

        self.travel_times_list = []
        self.episode_actions = dict()
    

    def _record(self, episode, ep_observations, start_time, agents):

        dc_episode, dc_ep_observations, dc_start_time, dc_agents = dc(episode), dc(ep_observations), dc(start_time), dc(agents)

        rewards = [{kc.AGENT_ID: agent.id, kc.REWARD: agent.last_reward} for agent in dc_agents]
        if (dc_episode in self.remember_episodes):
            self.recorder.record(dc_episode, dc_ep_observations, rewards)
        elif not self.frequent_progressbar:
            return
        msg = f"{self.phase_names[self.curr_phase]} {self.curr_phase+1}/{len(self.phases)}"
        curr_progress = dc_episode-self.phases[self.curr_phase]+1
        target = (self.phases[self.curr_phase+1]) if ((self.curr_phase+1) < len(self.phases)) else self.num_episodes+1
        target -= self.phases[self.curr_phase]
        #show_progress_bar(msg, dc_start_time, curr_progress, target)


    def _assign_rewards(self):
        for agent in self.all_agents:
            reward = agent.get_reward(self.travel_times_list)
            print(f"Agent {agent.id} reward: {reward}\n")

            # Add the reward in the travel_times_list
            for agent_entry in self.travel_times_list:
                if agent.id == agent_entry[kc.AGENT_ID]:
                    self.travel_times_list.remove(agent_entry)
                    agent_entry[kc.REWARD] = reward
                    self.travel_times_list.append(agent_entry)

            # Save machine's rewards based on PettingZoo standards
            if(agent.kind == 'AV'):
                self.rewards[str(agent.id)] = -1 * reward

            # Human learning
            elif self.human_learning == True:
                agent.learn(agent.last_action, self.travel_times_list)


    ###########################

    ##### Simulation loop #####

    def simulation_loop(self, machine_action, machine_id):
        """ This function contains the integration of the agent's actions to SUMO. 
        Description:
            We iterate through all the timesteps of the simulation.
            For each timestep there are None, one or more than one agents (humans, machines) that start. 
            If more than one machine agents have the same start time, we break from this function because we need to take the agent's action from the STEP function.
        Data structures:
            self.machine_same_start_time (list): contains the machine agents that their start time is equal to the simulator timestep
                and haven't acted yet.
            self.actions_timestep (list): includes the agents (machines/humans) that have acted in this timestep and
                their action will be send in the simulator
            agent_action (bool): break if the agent acting is not the last one (because the next agent should STEP first)
        """
        agent_action = False
        while self.simulator.timestep < self.simulation_params[kc.SIMULATION_TIMESTEPS] or len(self.travel_times_list) < len(self.all_agents):

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

                    # In case there are machine agents that have the same start time but it's not their turn
                    if (str(machine.id) != machine_id):

                        # If some machines have the same start time and they haven't acted yet
                        if (machine not in self.machine_same_start_time) and not any(machine == item[0] for item in self.actions_timestep):
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
                travel_times = self.help_step(self.actions_timestep)

                for agent_dict in travel_times:
                    self.travel_times_list.append(agent_dict)

                self.actions_timestep = []
                self.machine_same_start_time = []

            # If the machine agent that had turn acted
            if agent_action == True:
                agent_action = False
                break

    
    ###########################

    ##### Free flow times #####

    def get_free_flow_times(self):
        paths_df = pd.read_csv(self.simulator.paths_csv_path)
        origins = paths_df[kc.ORIGIN].unique()
        destinations = paths_df[kc.DESTINATION].unique()
        ff_dict = {(o, d): list() for o in origins for d in destinations}
        for _, row in paths_df.iterrows():
            ff_dict[(row[kc.ORIGIN], row[kc.DESTINATION])].append(row[kc.FREE_FLOW_TIME])
        return ff_dict
    
    ###########################

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    