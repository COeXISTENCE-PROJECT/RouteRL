from copy import copy
import functools
from gymnasium.spaces import Box, Discrete
import gymnasium as gym
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from pettingzoo.utils.env import ParallelEnv
import random
import pandas as pd
from keychain import Keychain as kc
from services import Simulator
from utilities import create_agent_objects
import seaborn as sns
import math


from keychain import Keychain as kc
from services.simulator import Simulator


## link https://pettingzoo.farama.org/tutorials/custom_environment/1-project-structure/
class TrafficEnvironment(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "TrafficEnvironment",
        "is_parallelizable": True,
    }

    def __init__(self, environment_params, simulation_params, agent_params, render_mode=None):

        self.simulator = Simulator(simulation_params)
        print("[SUCCESS] Environment initiated!")

        free_flows_dict = self.calculate_free_flow_times()
        print("\n[SUCCESS] Free flow times calculated!")


        self.simulation_params = simulation_params
        #self.possible_agents = ["1"]#, "2"] 
        self.possible_agents = [str(i) for i in range(1, 601)]

        self.agents = self.possible_agents

        self.observation_spaces = {
            agent: Box(low=0, high=1, shape=(1,), dtype=float) for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: gym.spaces.Discrete(simulation_params[kc.NUMBER_OF_PATHS]) for agent in self.possible_agents
        }

        ## save rewards and actions of each agent for the plots
        self.reward_table = {
            agent:[] for agent in self.possible_agents
        }

        self.action_table = {
            agent:[] for agent in self.possible_agents
        }

        ### Create start_time table
        num_origins = len(agent_params[kc.DESTINATIONS])
        num_destinations = len(agent_params[kc.DESTINATIONS])
        step_size = 1
        
        self.start_times = [i * step_size for i in range(len(self.possible_agents))]
        self.origin = [random.randrange(num_origins) for i in range(len(self.possible_agents))]
        self.destination = [random.randrange(num_destinations) for i in range(len(self.possible_agents))]

        ## Find the minimum free flow travel time from all the possible agents
        overall_min_travel_time = math.inf

        for i in range(len(self.origin)):
            print(f"\nAgent {i} has origin {self.origin[i]} and destination {self.destination[i]}.\n\n")

            travel_times = free_flows_dict[(self.origin[i], self.destination[i])]
            current_min_travel_time = min(travel_times)

            if (current_min_travel_time < overall_min_travel_time):
                overall_min_travel_time = current_min_travel_time

        self.overall_min_travel_time = overall_min_travel_time
        self.render_mode = render_mode



    def observe(self, agent):
        return Box(low=0, high=1, shape=(1,), dtype=float).sample() 

    def close(self):
        self.plot_rewards()
        self.plot_actions()
        

    def calculate_free_flow_times(self):
        free_flow_cost = self.simulator.calculate_free_flow_times()
        print('[INFO] Free-flow times: ', free_flow_cost)
        return free_flow_cost
    
    def start(self):
        self.simulator.start_sumo()
        state = None
        return state

    def stop(self):
        self.simulator.stop_sumo()
        state = None
        return state
        

    def reset(self, seed=None, options=None):
        self.simulator.reset_sumo()
        self.agents = copy(self.possible_agents)

        observations = {
            a: Box(low=0, high=1, shape=(1,), dtype=float).sample() for a in self.possible_agents
        }

        infos = {a: {}  for a in self.possible_agents}

        return observations, infos


    def step(self, joint_action):

        if not joint_action:
            self.possible_agents = []
            print("[INFO] No more agents to simulate!")
            return {}, {}, {}, {}, {}

        data = {
            'id': self.possible_agents,
            'action': [joint_action[agent] for agent in self.possible_agents],
            'origin': self.origin,
            'destination': self.destination,
            'start_time': self.start_times
        }

        # Create the DataFrame
        joint_action_df = pd.DataFrame(data)
        joint_action_df['id'] = joint_action_df['id'].astype(int)           

        ### Interact with SUMO to get travel times
        sumo_df = self.simulator.run_simulation_iteration(joint_action_df)
        sumo_df['id'] = sumo_df['id'].astype(str)

        
        ### Individual reward to each agent
        rewards = {}

        # Selfish agents
        """costs = sumo_df['travel_time'].values

        for agent_name in self.possible_agents:
            rewards[agent_name] = -1 * costs[i]"""

        #print(rewards)

        ### Joint reward for all agents
        joint_reward = self.calculate_rewards(sumo_df)
        #print("joint_reward is: ", joint_reward)

        for agent_name in self.possible_agents:
            rewards[agent_name] = joint_reward


        # Saves the actions and rewards of each agent for this episode
        for id, action in joint_action.items():
            self.action_table[id].append(action)
            self.reward_table[id].append(rewards[id])

        ### Return variables
        sample_observation = {
            a: (Box(low=0, high=1, shape=(1,), dtype=float).sample()) for a in self.possible_agents
        }

        terminated = {
            terminated: True for terminated in self.possible_agents
        }

        truncated = {
            truncated: 0 for truncated in self.possible_agents
        }

        info = {a: {} for a in self.possible_agents} 

        if any(terminated.values()) or all(truncated.values()):
            self.agents = []

        return sample_observation, rewards, terminated, truncated, info
    

    def plot_rewards(self):
        sns.set_style("whitegrid")

        random_agents = random.sample(self.possible_agents, 2)
        plt.figure(figsize=(20, 12)) 

        # Iterate over the selected agents and plot their rewards
        for agent_index in random_agents:
            plt.plot(self.reward_table[agent_index], linestyle='-', label=f'Agent {agent_index}')

        plt.xlabel('Episode', fontsize=12) 
        plt.ylabel('Reward', fontsize=12) 
        plt.title('Reward Table Over Episodes', fontsize=14) 
        plt.grid(True, linestyle='--', alpha=0.7)  
        plt.legend()
        plt.tight_layout() 
        plt.show()


    def plot_actions(self):
        sns.set_style("whitegrid")

        random_agents = random.sample(self.possible_agents, 5)
        plt.figure(figsize=(20, 12)) 

        # Iterate over the selected agents and plot their actions
        for agent_index in random_agents:
            plt.plot(self.action_table[agent_index], linestyle='-', label=f'Agent {agent_index}')

        plt.xlabel('Episode', fontsize=12) 
        plt.ylabel('Action', fontsize=12) 
        plt.title('Actions Over Episodes', fontsize=14)  
        plt.grid(True, linestyle='--', alpha=0.7)  
        plt.legend() 
        plt.tight_layout() 
        plt.show()


    def calculate_rewards(self, sumo_df):
        average_reward = -1 * sumo_df['travel_time'].mean() / self.overall_min_travel_time
        return average_reward


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(1,), dtype=float)


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.simulation_params[kc.NUMBER_OF_PATHS])
