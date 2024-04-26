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
import os
import numpy as np

from keychain import Keychain as kc
from services.simulator import Simulator
from services import SumoController


## link https://pettingzoo.farama.org/tutorials/custom_environment/1-project-structure/
class TrafficEnvironment(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "TrafficEnvironment",
        "is_parallelizable": True,
    }

    def __init__(self, environment_params, simulation_params, agent_params, nomachines, render_mode=None):

        self.simulation_params = simulation_params
        self.simulator = Simulator(simulation_params)
        print("[SUCCESS] Environment initiated!")

        self.free_flows_dict = self.calculate_free_flow_times()
        print("\n[SUCCESS] Free flow times calculated!")

        self.agent_params = agent_params
        self.simulation_params = simulation_params
        self.nomachines = nomachines
        self.render_mode = render_mode
        self.humans_learning = True

        if nomachines == False:

            ## Create the agents
            ## Machine agents
            self.possible_agents = [str(i) for i in range(1, 2)]
            self.n_agents = len(self.possible_agents)
            #self.possible_agents = [str(i) for i in range(0, agent_params[kc.NUM_AGENTS])]
        else:
            self.possible_agents = []
            self.n_agents = 0
            print("[INFO] There are no machines in this environment!")

        self.agents = self.possible_agents

        ## Human agents
        self.human_agents = create_agent_objects(agent_params, self.calculate_free_flow_times())

        self.initializeTheAgents(None)


    def say_hi(self):
        print("Hi from TrafficEnvironment!")

    def start(self):
        self.simulator.start_sumo()

    def stop(self):
        self.simulator.stop_sumo()

    
    def remake_sumo_controller(self):
        try:
            self.simulator.stop_sumo()
        except:
            print("[INFO] Sumo was not running!")
        self.simulator.sumo_controller = SumoController(self.simulation_params)
        print(f"New sumo label: {self.simulator.sumo_controller.label}")

        
        

    def initializeTheAgents(self, random_human):
        # Note: observations must be the same type as the weights of the NN, float32
        # The observation space represents the count of agents choosing each path with the same origin/destination pair as the current agent.
        self.observation_spaces = {
            agent: Box(low=0, high=self.agent_params[kc.NUM_AGENTS], shape=(3,), dtype=np.float32) for agent in self.possible_agents 
        }

        print("[INFO] Machine's observation space is:", self.observation_spaces, "\n\n")

        self.num_origins = len(self.agent_params[kc.DESTINATIONS])
        self.num_destinations = len(self.agent_params[kc.DESTINATIONS])
        self.num_paths = self.simulation_params[kc.NUMBER_OF_PATHS]

        """self.state_spaces = {
            Box(low=0, high=self.agent_params[kc.NUM_AGENTS], shape=(self.num_origins, self.num_destinations, self.num_paths), dtype=np.float32) 
        }"""
        
        self.action_spaces = {
            agent: gym.spaces.Discrete(self.simulation_params[kc.NUMBER_OF_PATHS]) for agent in self.possible_agents
        }

        print("[INFO] Machine's action space is:", self.action_spaces, "\n\n")

        ## Save rewards and actions of each agent for the plots
        self.reward_table = {
            agent:[] for agent in self.possible_agents
        }

        self.action_table = {
            agent:[] for agent in self.possible_agents
        }

        self.action_table_humans = {
            agent.id:[] for agent in self.human_agents
        }

        self.reward_table_humans = {
            agent.id:[] for agent in self.human_agents
        }

        ### Create start_time table
        step_size = 6
        
        #self.start_times = [i * step_size for i in range(len(self.possible_agents))]
        self.start_times = [random_human.start_time for i in range(len(self.possible_agents))]
        self.origin = [random_human.origin for i in range(len(self.possible_agents))]
        self.destination = [random_human.destination for i in range(len(self.possible_agents))]
        """self.origin = [random.randrange(self.num_origins) for i in range(len(self.possible_agents))]
        self.destination = [random.randrange(self.num_destinations) for i in range(len(self.possible_agents))]"""

        for agent in self.possible_agents:
            print("Agent with id", agent, "has origin", self.origin[int(agent) - 1], "and destination", self.destination[int(agent) - 1], "and start time", self.start_times[int(agent) - 1])
        
        self.min_reward = - math.inf

        ## Divide all travel times with this value -> Normalization
        self.overall_min_travel_time = self.min_travel_time()

        print("[INFO] Minimum travel time is: ", self.overall_min_travel_time)


    def state(self):
        matrix_shape = (self.num_origins, self.num_destinations, self.num_paths)
        agent_counts = np.zeros(matrix_shape, dtype=int)

        # Iterate over agents and observations
        for agent in self.human_agents:
            # Determine the action chosen by the agent (assuming it's a number between 0 and num_paths-1)
            action = agent.act(0)
            
            # Increment the count for the corresponding path in the matrix
            agent_counts[agent.origin][agent.destination][action] += 1

        # Print the resulting matrix
        #print("[INFO] state was used: \n", agent_counts)

        return agent_counts


    def reset(self, seed=None, options=None):
        print("[INFO] RESET")
        self.remake_sumo_controller()
        self.simulator.start_sumo()
        self.agents = copy(self.possible_agents)

        observations = {
            a: np.zeros(3, dtype=np.float32) for a in self.possible_agents
        }

        infos = {a: {}  for a in self.possible_agents}

        return observations, infos



    def step(self, machine_joint_action):
        print("[INFO] STEP")
        self.remake_sumo_controller()
        self.simulator.start_sumo()
        if not machine_joint_action and not self.human_agents:
            self.possible_agents = []
            print("[INFO] No more agents to simulate!")
            return {}, {}, {}, {}, {}
                
        ## Preprocess the human and machine joint actions
        joint_action_df, human_joint_action, state_table = self.prepare_joint_action(machine_joint_action)

        ## Interact with SUMO to get travel times
        sumo_df = self.simulator.run_simulation_iteration(joint_action_df)

        ## Human learning
        self.human_learning(sumo_df, human_joint_action)

        ## Return observations
        if self.nomachines == False:
            observations = self.return_observation()

            ## Machine learning
            rewards, terminated, truncated, info = self.machine_learning(sumo_df, machine_joint_action, state_table)

            return observations, rewards, terminated, truncated, info

        else:
            observations = {}        


    def min_travel_time(self):

        ## Find the minimum free flow travel time from all the possible agents
        overall_min_travel_time = math.inf

        for agent in self.human_agents:
            travel_times = self.free_flows_dict[(agent.origin, agent.destination)]
            current_min_travel_time = min(travel_times)

            if (current_min_travel_time < overall_min_travel_time):
                overall_min_travel_time = current_min_travel_time

        return overall_min_travel_time
        



    def observe(self, agent):
        return Box(low=0, high=self.agent_params[kc.NUM_AGENTS], shape=(3,), dtype=int)
    

    def close(self):
        print("[INFO] CLOSE")
        self.simulator.reset_sumo()
        self.plot_rewards()
        #self.plot_actions()

        ## Save the minimum rewards observed in every environment
        file_path = "min_reward.txt"

        """with open(file_path, 'a') as file:
            file.write(str(self.min_reward) + '\n')"""

        ## Empty the reward and action tables
        self.reward_table = {
            agent:[] for agent in self.possible_agents
        }

        self.action_table = {
            agent:[] for agent in self.possible_agents
        }

        self.action_table_humans = {
            agent.id:[] for agent in self.human_agents
        }

        self.reward_table_humans = {
            agent.id:[] for agent in self.human_agents
        }

    def return_observation(self):
        ### Works for one agent - if more agents needs adjustment

        matrix_shape = (self.num_paths)
        agent_counts = np.zeros(matrix_shape, dtype=int)
        # Iterate over agents and observations
        for agent in self.human_agents:

            if(agent.origin == self.origin[0] and agent.destination == self.destination[0]):
                action = agent.act(0)
                agent_counts[action] += 1

        observations = {
            agents: agent_counts for agents in self.possible_agents
        }

        return observations
        

    def calculate_free_flow_times(self):
        free_flow_cost = self.simulator.calculate_free_flow_times()
        print('[INFO] Free-flow times: ', free_flow_cost)

        return free_flow_cost
    
    
    def human_actions(self):
        ## Human agent's action
        observations = {
            a: Box(low=0, high=self.agent_params[kc.NUM_AGENTS], shape=(3,), dtype=np.float32).sample() for a in self.human_agents
        }

        human_joint_action = self.get_human_joint_action(self.human_agents, observations)

        ## Change the id numbers so they don't collide with machine agents
        human_joint_action['id'] = human_joint_action['id'] + self.agent_params[kc.NUM_AGENTS] 

        ## Remove the kind column
        human_joint_action.drop(columns=['kind'], inplace=True) 

        return human_joint_action
    
    
    def machine_learning(self, sumo_df, machine_joint_action, state_table):
        print("[INFO] Machines are about to learn!")

        sumo_df['id'] = sumo_df['id'].astype(str)
        sumo_df_machines = sumo_df.head(self.agent_params[kc.NUM_AGENTS])
        state_table = state_table.head(self.agent_params[kc.NUM_AGENTS])
        
        ## Individual reward to each machine agent
        rewards = {}

        ## Joint reward for all machine agents
        joint_reward = self.calculate_rewards(sumo_df_machines)

        for agent_name in self.possible_agents:
            rewards[agent_name] = joint_reward

        ## Saves the actions and rewards of each agent for this episode
        for id, action in machine_joint_action.items():
            self.action_table[id].append(action)
            self.reward_table[id].append(rewards[id])


        ## Return variables
        terminated = {
            terminated: True for terminated in self.possible_agents
        }

        truncated = {
            truncated: 0 for truncated in self.possible_agents
        }

        info = {a: {} for a in self.possible_agents} 

        if any(terminated.values()) or all(truncated.values()):
            self.agents = []

        return rewards, terminated, truncated, info
    
    
    
    def prepare_joint_action(self, machine_joint_action):
        data = {
            'id': self.possible_agents,
            'action': [machine_joint_action[agent] for agent in self.possible_agents],
            'origin': self.origin,
            'destination': self.destination,
            'start_time': self.start_times
        }

        ## Human agent's action
        human_joint_action = self.human_actions()


        # Create the DataFrame
        joint_action_df = pd.DataFrame(data)
        joint_action_df = joint_action_df.astype(int)


        ## Combine human agents with machine agents
        joint_action_df = pd.concat([joint_action_df, human_joint_action], ignore_index=True)  

        ## Create a dataframe that will contain the state table
        # Group by 'origin', 'destination', and 'action', and count the occurrences
        action_counts = joint_action_df.groupby(['origin', 'destination', 'action']).size().reset_index(name='count')

        # Filter action_counts based on the maximum action value
        action_counts = action_counts[action_counts['action'] < self.simulation_params[kc.NUMBER_OF_PATHS] + 1]

        # Pivot the table to have 'origin' and 'destination' as rows, 'action' as columns, and 'count' as values
        state_table = action_counts.pivot_table(index=['origin', 'destination'], columns='action', values='count', fill_value=0)

        state_table.reset_index(inplace=True)
        state_table = state_table.astype(int)
        state_table = pd.merge(joint_action_df, state_table, on=['origin', 'destination'], how='inner')


        return joint_action_df, human_joint_action, state_table
    
    def mutation(self):
        print("[INFO] Mutation is about to happen!\n")

        print("[INFO] There were", len(self.human_agents), "human agents.\n")
        random_human = random.choice(self.human_agents)
        self.human_agents.remove(random_human)
        print("[INFO] Now there are", len(self.human_agents), "human agents.\n")

        self.possible_agents = [str(i) for i in range(1, 2)]
        self.n_agents = len(self.possible_agents)
        self.nomachines = False
        self.humans_learning = False

        self.initializeTheAgents(random_human)
    
 
    def human_learning(self, sumo_df, human_joint_action):

        ## Separate the human agents from the machine agents
        split_value = self.agent_params[kc.NUM_AGENTS]

        ## Keep the human agents
        human_df = sumo_df[sumo_df['id'] >= split_value]
        human_joint_action = human_joint_action[human_joint_action['id'] >= split_value]

        human_df.loc[:, 'id'] -= split_value
        human_joint_action.loc[:, 'id'] -= split_value

        ## Human agents
        for agent in self.human_agents:
            
            action = human_joint_action.loc[human_joint_action[kc.AGENT_ID] == agent.id, kc.ACTION].item()
            reward = -1 * human_df.loc[human_df[kc.AGENT_ID] == agent.id, "travel_time"].item() / self.overall_min_travel_time

            self.action_table_humans[agent.id].append(action)
            self.reward_table_humans[agent.id].append(reward)

            if self.human_learning == True:
                print("[INFO] Humans are about to learn!")
                observation = 0
                agent.learn(action, reward, observation)    


    def plot_rewards(self):
        sns.set_style("whitegrid")

        ## Choose 1 random agent and plot its rewards
        if(self.possible_agents != []):
            random_agents = random.sample(self.possible_agents, 1)
        else:
            random_human_agent = random.choice(self.human_agents)
            random_agents = random_human_agent.id
            random_agents = [random_agents]

        plt.figure(figsize=(20, 12)) 

        ## Save the plot in the results folder
        file_number = 1
        while os.path.exists(f"results/rewards{file_number}.png"):
            file_number += 1

        filename = f"results/rewards{file_number}.png"


        # Iterate over the selected agents and plot their rewards
        for agent_index in random_agents:
            if(self.possible_agents!= []):
                plt.plot(self.reward_table[agent_index], linestyle='-', label=f'Machine Agent {agent_index}')
            plt.plot(self.reward_table_humans[int(agent_index)], linestyle='-', label=f'Human Agent {agent_index}')

        plt.xlabel('Episode', fontsize=12) 
        plt.ylabel('Reward', fontsize=12) 
        plt.title(f'Reward Table Over Episodes for {self.agent_params[kc.NUM_AGENTS]} agents', fontsize=14) 
        plt.grid(True, linestyle='--', alpha=0.7)  
        plt.legend()
        plt.tight_layout() 
        plt.savefig(filename)
        plt.show()


    def plot_actions(self):
        sns.set_style("whitegrid")

        ## Choose 2 random agents and plot their actions
        if(len(self.possible_agents) < 2):
            random_agents = self.possible_agents
        else:
            random_agents = random.sample(self.possible_agents, 3)

        ## Save the plot in the results folder
        file_number = 1
        while os.path.exists(f"results/actions{file_number}.png"):
            file_number += 1

        filename = f"results/actions{file_number}.png"

        plt.figure(figsize=(20, 12)) 

        ## Iterate over the selected agents and plot their actions
        for agent_index in random_agents:
            plt.plot(self.action_table[agent_index], linestyle='-', label=f'Agent {agent_index}')
            plt.plot(self.action_table_humans[int(agent_index)], linestyle='-', label=f'Agent {agent_index}')

        plt.xlabel('Episode', fontsize=12) 
        plt.ylabel('Action', fontsize=12) 
        plt.title(f'Actions Over Episodes for {self.agent_params[kc.NUM_AGENTS]} agents', fontsize=14)  
        plt.grid(True, linestyle='--', alpha=0.7)  
        plt.legend() 
        plt.tight_layout() 
        plt.savefig(filename)
        plt.show()

        
    def get_human_joint_action(self, agents, observations):
        joint_action_cols = [kc.AGENT_ID, kc.AGENT_KIND, kc.ACTION, kc.AGENT_ORIGIN, kc.AGENT_DESTINATION, kc.AGENT_START_TIME]
        joint_action = pd.DataFrame(columns = joint_action_cols)

        # Every agent picks action
        for agent, observation in zip(agents, observations):
            action = agent.act(observation)
            action_data = [agent.id, agent.kind, action, agent.origin, agent.destination, agent.start_time]
            joint_action.loc[len(joint_action.index)] = {key : value for key, value in zip(joint_action_cols, action_data)}
        return joint_action   


    def calculate_rewards(self, sumo_df):
        average_reward = -1 * sumo_df['travel_time'].mean() / self.overall_min_travel_time

        if average_reward > self.min_reward:
            self.min_reward = average_reward

        return average_reward


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=self.agent_params[kc.NUM_AGENTS], shape=(self.simulation_params[kc.NUMBER_OF_PATHS],), dtype=np.float32)


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.simulation_params[kc.NUMBER_OF_PATHS])
