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
        self.actions = []
        print("[SUCCESS] Environment initiated!")
        free_flows_dict = self.calculate_free_flow_times()
        print("[SUCCESS] Free flow times calculated!")
        
        self.possible_agents = ["1"] 
        #self.possible_agents = [str(i) for i in range(1, 601)]

        self.agents = self.possible_agents

        self.observation_spaces = {
            agent: Box(low=0, high=1, shape=(1,), dtype=float) for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: gym.spaces.Discrete(3) for agent in self.possible_agents
        }

        ### Create start_time table
        num_origins = len(agent_params[kc.DESTINATIONS])
        num_destinations = len(agent_params[kc.DESTINATIONS])
        step_size = 6
        
        self.start_times = [i * step_size for i in range(len(self.possible_agents))]
        self.origin = [random.randrange(num_origins) for i in range(len(self.possible_agents))]
        self.destination = [random.randrange(num_destinations) for i in range(len(self.possible_agents))]

        print("[SUCCESS]: The vehicle will travel from origin ", self.origin, " to destination.", self.destination, " This path has free flow travel time: ", free_flows_dict[(self.origin[0], self.destination[0])])

        """print("self.origin is: ", self.origin, "\n\n")
        print("self.destination is: ", self.destination, "\n\n")
        print("self.start_times is: ", self.start_times, "\n\n")"""

        self.render_mode = render_mode

        """self.od_pairs = []
        number_of_agents = len(self.possible_agents)

        for i in range(number_of_agents):
            if i < number_of_agents // 2:
                self.od_pairs.append("0_0")
            else:
                self.od_pairs.append("1_1")

        print("[INFO] OD pairs: ", self.od_pairs)"""



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
        print("Actions are: ", self.actions)
        self.plot_rewards()
        self.plot_actions()
        self.reward_table = []
        self.actions = []
        


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
            

            if(i == 500):
                self.reward_table.append(-1 * costs[i])

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
        plt.figure(figsize=(10, 6)) 
        plt.plot(self.reward_table, color='blue', linestyle='-')  
        plt.xlabel('Episode', fontsize=12) 
        plt.ylabel('Reward', fontsize=12) 
        plt.title('Reward Table Over Episodes', fontsize=14)  
        plt.grid(True, linestyle='--', alpha=0.7)  
        plt.tight_layout() 
        plt.show()

    def plot_actions(self):
        plt.figure(figsize=(10, 6)) 
        plt.plot(self.actions, color='blue', linestyle='-')  
        plt.xlabel('Episode', fontsize=12) 
        plt.ylabel('Action', fontsize=12) 
        plt.title('Actions Over Episodes', fontsize=14)  
        plt.grid(True, linestyle='--', alpha=0.7)  
        plt.tight_layout() 
        plt.show()

    def calculate_rewards(self, sumo_df):
        average_reward = -1 * sumo_df['travel_time'].mean()
        self.reward_table.append(average_reward)
        return average_reward


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(1,), dtype=float)


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)
