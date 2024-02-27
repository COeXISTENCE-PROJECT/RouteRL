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
        self.reward_table = []
        #self.reward_table2 = []
        self.actions = []
        #self.actions2 = []
        print("[SUCCESS] Environment initiated!")
        free_flows_dict = self.calculate_free_flow_times()
        print("[SUCCESS] Free flow times calculated!")


        self.simulation_params = simulation_params
        self.possible_agents = ["1"]#, "2"] 
        #self.possible_agents = [str(i) for i in range(1, 601)]

        self.agents = self.possible_agents

        self.observation_spaces = {
            agent: Box(low=0, high=1, shape=(1,), dtype=float) for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: gym.spaces.Discrete(simulation_params[kc.NUMBER_OF_PATHS]) for agent in self.possible_agents
        }

        ### Create start_time table
        num_origins = len(agent_params[kc.DESTINATIONS])
        num_destinations = len(agent_params[kc.DESTINATIONS])
        step_size = 6
        
        self.start_times = [i * step_size for i in range(len(self.possible_agents))]
        self.origin = [random.randrange(num_origins) for i in range(len(self.possible_agents))]
        self.destination = [random.randrange(num_destinations) for i in range(len(self.possible_agents))]

        for i in range(len(self.origin)):
            print(f"Agent {i} has origin {self.origin[i]} and destination {self.destination[i]}.")

        self.render_mode = render_mode



    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        return Box(low=0, high=1, shape=(1,), dtype=float).sample() 

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        print("Reward table is: ", self.reward_table)
        #print("Reward table is: ", self.reward_table2)
        print("Actions are: ", self.actions)
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

        self.actions.append(joint_action['1'])
        #self.actions2.append(joint_action['2'])


        # Create the DataFrame
        joint_action_df = pd.DataFrame(data)
        joint_action_df['id'] = joint_action_df['id'].astype(int)
        

        ### Interact with SUMO to get travel times
        sumo_df = self.simulator.run_simulation_iteration(joint_action_df)
        sumo_df['id'] = sumo_df['id'].astype(str)

        
        costs = sumo_df['travel_time'].values


        ### Individual reward to each agent
        rewards = {}

        # each agent tries to minimize each one travel time
        """i = 0
        for agent_name in self.possible_agents:
            rewards[agent_name] = -1 * costs[i]
            

            if(i == 0):
                self.reward_table.append(-1 * costs[i])
            else:
                self.reward_table2.append(-1 * costs[i])

            i = i + 1"""

        #print(rewards)

        ### Joint reward for all agents
        joint_reward = self.calculate_rewards(sumo_df)

        for agent_name in self.possible_agents:
            rewards[agent_name] = joint_reward

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

        plt.figure(figsize=(20, 12)) 
        plt.plot(self.reward_table, color='blue', linestyle='-')  
        #plt.plot(self.reward_table2, color='red', linestyle='-')  
        plt.xlabel('Episode', fontsize=12) 
        plt.ylabel('Reward', fontsize=12) 
        plt.title('Reward Table Over Episodes', fontsize=14)  
        plt.tight_layout() 
        plt.show()

        """num_plots = len(self.actions) // 1000
        remainder = len(self.actions) % 1000

        if remainder > 0:
            num_plots += 1

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3*num_plots))

        for i in range(num_plots):
            start_index = i * 1000
            end_index = min(start_index + 1000, len(self.actions))
            ax = axes[i] if num_plots > 1 else axes

            ax.plot(self.reward_table[start_index:end_index], color='blue', linestyle='-', label=f'Actions {i+1}')
            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Reward', fontsize=12)
            ax.set_title(f'Rewards Over Episodes (Plot {i+1}, Learning rate {0.1*i})', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            plt.tight_layout()

        plt.show()"""

    def plot_actions(self):
        sns.set_style("whitegrid")

        plt.figure(figsize=(20, 12)) 
        plt.plot(self.actions, color='blue', linestyle='-', label='Actions 1')  
        #plt.plot(self.actions2, color='red', linestyle='-', label='Actions 2')  # Plot actions2
        plt.xlabel('Episode', fontsize=12) 
        plt.ylabel('Action', fontsize=12) 
        plt.title('Actions Over Episodes', fontsize=14)  
        plt.grid(True, linestyle='--', alpha=0.7)  
        plt.legend()  # Show legend to differentiate between Actions 1 and Actions 2
        plt.tight_layout() 
        plt.show()

        """num_plots = len(self.actions) // 1000
        remainder = len(self.actions) % 1000

        if remainder > 0:
            num_plots += 1

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3*num_plots))

        for i in range(num_plots):
            start_index = i * 1000
            end_index = min(start_index + 1000, len(self.actions))
            ax = axes[i] if num_plots > 1 else axes

            ax.plot(self.actions[start_index:end_index], color='blue', linestyle='-', label=f'Actions {i+1}')
            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Action', fontsize=12)
            ax.set_title(f'Actions Over Episodes (Plot {i+1}, Learning rate {0.1*i})', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            plt.tight_layout()

        plt.show()"""
    


    def calculate_rewards(self, sumo_df):
        average_reward = -1 * sumo_df['travel_time'].mean()
        self.reward_table.append(average_reward)
        return average_reward


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(1,), dtype=float)


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.simulation_params[kc.NUMBER_OF_PATHS])
