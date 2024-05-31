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
import logging
import scienceplots

# Configure logging to show messages of all levels
logging.basicConfig(level=logging.DEBUG)
#plt.style.use('science')



## link https://pettingzoo.farama.org/tutorials/custom_environment/1-project-structure/
class TrafficEnvironment(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "TrafficEnvironment",
        "is_parallelizable": True,
    }

    def __init__(self, training_params, environment_params, simulation_params, agent_params, nomachines, render_mode=None):

        self.simulation_params = simulation_params
        self.simulator = Simulator(simulation_params)
        logging.info("Environment initiated!")

        self.free_flows_dict = self.calculate_free_flow_times()
        logging.info("Free flow times calculated!")

        self.agent_params = agent_params
        self.simulation_params = simulation_params
        self.training_params = training_params
        self.nomachines = nomachines
        self.render_mode = render_mode
        self.humans_learning = True
        self.human_joint_action = pd.DataFrame()

        if nomachines == False:

            ## Create the agents
            ## Machine agents
            self.possible_agents = [str(i) for i in range(1, 2)]
            self.n_agents = len(self.possible_agents)
            #self.possible_agents = [str(i) for i in range(0, agent_params[kc.NUM_AGENTS])]
        else:
            self.possible_agents = []
            self.n_agents = 0
            logging.info("There are no machines in this environment!")

        self.agents = self.possible_agents

        ## Human agents
        self.human_agents = create_agent_objects(agent_params, self.calculate_free_flow_times())

        self.action_table_humans = {
            agent.id:[] for agent in self.human_agents
        }

        self.reward_table_humans = {
            agent.id:[] for agent in self.human_agents
        }

        self.initializeTheAgents(None)


    def say_hi(self):
        logging.info("Hi from TrafficEnvironment!")

    def start(self):
        self.simulator.start_sumo()

    def stop(self):
        self.simulator.stop_sumo()

    
    def remake_sumo_controller(self):
        try:
            self.simulator.stop_sumo()
        except:
            logging.error("Sumo was not running!")
        self.simulator.sumo_controller = SumoController(self.simulation_params)
        #logging.info(f"New sumo label: {self.simulator.sumo_controller.label}")


    def initializeTheAgents(self, random_human):
        # Note: observations must be the same type as the weights of the NN, float32
        # The observation space represents the count of agents choosing each path with the same origin/destination pair as the current agent.
        self.observation_spaces = {
            agent: Box(low=0, high=self.agent_params[kc.NUM_HUMAN_AGENTS], shape=(3,), dtype=np.float32) for agent in self.possible_agents 
        }

        logging.info("Machine's observation space is: %s ", self.observation_spaces)

        self.num_origins = len(self.agent_params[kc.DESTINATIONS])
        self.num_destinations = len(self.agent_params[kc.DESTINATIONS])
        self.num_paths = self.simulation_params[kc.NUMBER_OF_PATHS]

        self.action_spaces = {
            agent: gym.spaces.Discrete(self.simulation_params[kc.NUMBER_OF_PATHS]) for agent in self.possible_agents
        }

        logging.info("Machine's action space is: %s", self.action_spaces)

        ## Save rewards and actions of each agent for the plots
        
        self.reward_table = {
            agent:[] for agent in self.possible_agents
        }

        self.action_table = {
            agent:[] for agent in self.possible_agents
        }

        
        self.start_times = [random_human.start_time for i in range(len(self.possible_agents))]
        self.origin = [random_human.origin for i in range(len(self.possible_agents))]
        self.destination = [random_human.destination for i in range(len(self.possible_agents))]
        """self.origin = [random.randrange(self.num_origins) for i in range(len(self.possible_agents))]
        self.destination = [random.randrange(self.num_destinations) for i in range(len(self.possible_agents))]"""

        for agent in self.possible_agents:
            logging.info("Agent with id %s has origin %s and destination %s and start time %s",
             agent, self.origin[int(agent) - 1], self.destination[int(agent) - 1],
             self.start_times[int(agent) - 1])



        self.min_reward = - math.inf

        ## Divide all travel times with this value -> Normalization
        self.overall_min_travel_time = self.min_travel_time()

        logging.info("Minimum travel time is: %s", self.overall_min_travel_time)


    def state(self):
        matrix_shape = (self.num_origins, self.num_destinations, self.num_paths)
        agent_counts = np.zeros(matrix_shape, dtype=int)

        # Iterate over agents and observations
        for agent in self.human_agents:
            # Determine the action chosen by the agent (assuming it's a number between 0 and num_paths-1)
            action = agent.act(0)
            
            # Increment the count for the corresponding path in the matrix
            agent_counts[agent.origin][agent.destination][action] += 1

        return agent_counts


    def reset(self, seed=None, options=None):
        #logging.info("RESET")
        self.remake_sumo_controller()
        self.simulator.start_sumo()
        self.agents = copy(self.possible_agents)

        observations = {
            a: np.zeros(3, dtype=np.float32) for a in self.possible_agents
        }

        infos = {a: {}  for a in self.possible_agents}

        return observations, infos



    def step(self, machine_joint_action):
        #logging.info("STEP")
        self.remake_sumo_controller()
        self.simulator.start_sumo()

        if not machine_joint_action and not self.human_agents:
            self.possible_agents = []
            logging.info("No more agents to simulate!")
            return {}, {}, {}, {}, {}
        
        ## Preprocess the human and machine joint actions
        joint_action_df, human_joint_action, state_table = self.prepare_joint_action(machine_joint_action)

        ## Interact with SUMO to get travel times
        sumo_df = self.simulator.run_simulation_iteration(joint_action_df)

        ## Human learning
        self.human_learning(sumo_df, human_joint_action)

        ## Prepare the next action that will be taken by the human agents -> so it can be passed as observation  machine's learning
        self.human_joint_action = self.human_actions()

        ## Return observations
        if self.nomachines == False:
            
            ## Return observations
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
        logging.info("CLOSE")
        self.simulator.reset_sumo()
        self.plot_rewards()
        self.plot_actions()

        ## Save the minimum rewards observed in every environment
        file_path = "min_reward.txt"

        """with open(file_path, 'a') as file:
            file.write(str(self.min_reward) + '\n')"""
        

    def return_observation(self):
        matrix_shape = (self.num_paths,)
        agent_counts = np.zeros(matrix_shape, dtype=int)
        num_agents_same_od = 0

        for index, row in self.human_joint_action.iterrows():
            if (row['origin'] == self.origin[0] and 
                row['destination'] == self.destination[0] and 
                row['start_time'] <= self.start_times[0]):
                
                num_agents_same_od += 1
                action = row['action']
                agent_counts[action] += 1

        observations = {
            ## normalize the observatio table
            agents: agent_counts/num_agents_same_od for agents in self.possible_agents
        }

        #print("Observations are:", observations, '\n')

        return observations
        

    def calculate_free_flow_times(self):
        free_flow_cost = self.simulator.calculate_free_flow_times()
        logging.info('Free-flow times: %s', free_flow_cost)

        return free_flow_cost
    
    
    def human_actions(self):
        ## Human agent's action
        observations = {
            a: Box(low=0, high=self.agent_params[kc.NUM_HUMAN_AGENTS], shape=(3,), dtype=np.float32).sample() for a in self.human_agents
        }

        joint_action_cols = [kc.AGENT_ID, kc.AGENT_KIND, kc.ACTION, kc.AGENT_ORIGIN, kc.AGENT_DESTINATION, kc.AGENT_START_TIME]
        human_joint_action = pd.DataFrame(columns = joint_action_cols)

        # Every agent picks action
        for agent, observation in zip(self.human_agents, observations):
            action = agent.act(observation)
            action_data = [agent.id, agent.kind, action, agent.origin, agent.destination, agent.start_time]
            human_joint_action.loc[len(human_joint_action.index)] = {key : value for key, value in zip(joint_action_cols, action_data)}

        ## Change the id numbers so they don't collide with machine agents
        human_joint_action['id'] = human_joint_action['id'] + self.agent_params[kc.NUM_HUMAN_AGENTS] 

        ## Remove the kind column
        human_joint_action.drop(columns=['kind'], inplace=True) 

        return human_joint_action
    
    
    def machine_learning(self, sumo_df, machine_joint_action, state_table):
        #logging.info("Machines are about to learn!")

        sumo_df['id'] = sumo_df['id'].astype(str)
        sumo_df_machines = sumo_df.head(self.agent_params[kc.NUM_MACHINE_AGENTS])
        state_table = state_table.head(self.agent_params[kc.NUM_MACHINE_AGENTS])
        
        ## Individual reward to each machine agent
        rewards = {}

        ## Calculate the rewards for each machine agent
        for index, row in sumo_df_machines.iterrows():
            if row['id'] in self.possible_agents:
                rewards[row['id']] = -1 * row['travel_time'] #/ self.overall_min_travel_time

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
        ## If we are in the first iteration and human_joint_action is empty
        if(self.human_joint_action.empty == True):
            self.human_joint_action = self.human_actions()

        # Create the DataFrame
        joint_action_df = pd.DataFrame(data)
        joint_action_df = joint_action_df.astype(int)


        ## Combine human agents with machine agents
        joint_action_df = pd.concat([joint_action_df, self.human_joint_action], ignore_index=True)  

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


        return joint_action_df, self.human_joint_action, state_table
    
    def mutation(self):

        logging.info("Mutation is about to happen!\n")
        logging.info("There were %s human agents.\n", len(self.human_agents))

        ### Mutate to a human that starts after the 25% of the rest of the vehicles

        # Calculate the 25th percentile of the start_time values
        start_times = [human.start_time for human in self.human_agents]
        percentile_25 = np.percentile(start_times, 25)

        # Filter the human agents whose start_time is higher than the 25th percentile
        filtered_human_agents = [human for human in self.human_agents if human.start_time > percentile_25]

        # Randomly select one of the filtered human agents
        random_human = random.choice(filtered_human_agents)
        print(f"Selected human agent with start time: {random_human.start_time}")


        self.human_agents.remove(random_human)
        logging.info("Now there are %s human agents.\n", len(self.human_agents))

        self.table_before_mutation = self.reward_table_humans
        #self.empty_reward_action_tables()
        print("self.table_before_mutation is: ", self.table_before_mutation, "\n\n\n")

        self.possible_agents = [str(i) for i in range(1, 2)]
        self.n_agents = len(self.possible_agents)
        self.nomachines = False
        self.humans_learning = False

        self.initializeTheAgents(random_human)
    
 
    def human_learning(self, sumo_df, human_joint_action):

        ## Separate the human agents from the machine agents
        split_value = self.agent_params[kc.NUM_HUMAN_AGENTS]

        ## Keep the human agents
        human_df = sumo_df[sumo_df['id'] >= split_value]
        human_joint_action = human_joint_action[human_joint_action['id'] >= split_value]

        human_df.loc[:, 'id'] -= split_value
        human_joint_action.loc[:, 'id'] -= split_value

        ## Human agents
        for agent in self.human_agents:
            
            action = human_joint_action.loc[human_joint_action[kc.AGENT_ID] == agent.id, kc.ACTION].item()
            reward = -1 * human_df.loc[human_df[kc.AGENT_ID] == agent.id, "travel_time"].item() #/ self.overall_min_travel_time

            self.action_table_humans[agent.id].append(action)
            self.reward_table_humans[agent.id].append(reward)

            if self.human_learning == True:
                logging.info("Humans are about to learn!")
                observation = 0
                agent.learn(action, reward, observation)  

    def compare_machine_human(self):
        ## Compare rewards of different vehicles
        listofhumans = []

        for index, row in self.human_joint_action.iterrows():

            if(row['origin'] == self.origin[0] and row['destination'] == self.destination[0]):
                
                ## We want to calculate the average reward after the mutation episode
                human_avg = sum(self.reward_table_humans[index][self.training_params[kc.HUMAN_LEARNING_LENGTH]:]) / (len(self.reward_table_humans[index]) - self.training_params[kc.HUMAN_LEARNING_LENGTH])
                listofhumans.append(human_avg)

        print("list of humans is: ", listofhumans, "\n\n\n")

        # Reference reward from self.reward_table
        reference_reward = sum(self.reward_table['1']) / len(self.reward_table['1'])
        print("reference reward is: ", reference_reward, "\n\n\n")

        # Count the number of rewards in listofhumans that are less than the reference_reward
        count_lower = sum(1 for reward in listofhumans if reward < reference_reward)

        # Calculate the percentage of rewards lower than the reference_reward
        percentage_lower = (count_lower / len(listofhumans)) * 100

        print("Percentage of rewards lower than the reference value: ", percentage_lower, "%")

        # Format the percentage as a string
        percentage_str = f"\n{percentage_lower}%"

        # Specify the filename
        filename = "percentage.txt"

        # Open the file in write mode and save the percentage
        with open(filename, "a") as file:
            file.write(percentage_str)

        print(f"The percentage {percentage_str} has been saved to {filename}")
        


    def plot_rewards(self):
        sns.set_style("whitegrid")

        mutation_episode = self.training_params[kc.HUMAN_LEARNING_LENGTH]
        print("mutation episode is: ", mutation_episode, "\n\n\n")
        print("self.human_reward_table is: ", self.reward_table_humans, len(self.reward_table_humans[0]), "\n\n\n")

        if self.possible_agents:
            random_agents = random.sample(self.possible_agents, 1)
            human_agents = [agent for agent in self.human_agents if agent.origin == self.origin[0] and agent.destination == self.destination[0]]
            human_agents = random.sample(human_agents, 2)
        else:
            human_agents = random.sample(self.human_agents, 2)

        plt.figure(figsize=(20, 12), dpi=200) 

        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
        for idx, human in enumerate(human_agents):
            color = colors[idx]
            plt.plot(self.reward_table_humans[human.id], linestyle='-', color=color, linewidth=3, label=f'Human Agent {human.id} with o-d pair: {human.origin} - {human.destination}')

        if self.possible_agents:
            for agent_index in random_agents:
                if self.possible_agents:
                    for idx, human in enumerate(human_agents):
                        color = colors[idx]
                        x_values = [mutation_episode, mutation_episode + 1]
                        y_values = [self.reward_table_humans[human.id][-1], self.reward_table_humans[human.id][0]]
                        plt.plot(x_values, y_values, linestyle='-', color=color, linewidth=3)
                        plt.plot(self.reward_table_humans[human.id], linestyle='-', color=color, linewidth=3)
                    plt.plot(np.arange(mutation_episode, mutation_episode + len(self.reward_table[agent_index])), self.reward_table[agent_index], linestyle='-', linewidth=3, color='tab:green', label=f'Machine Agent with origin-destination pair: {self.origin[0]} - {self.destination[0]}')
                else:
                    plt.plot(self.reward_table_humans[int(agent_index)], linestyle='-', linewidth=3)

            plt.axvline(x=mutation_episode, color='k', linestyle='--', label='Mutation Time', linewidth=3)

        plt.tick_params(axis='both', which='major', labelsize=25)
        plt.xlabel('Episode', fontsize=25)
        plt.ylabel('Reward', fontsize=25)
        #plt.title(f'Rewards Over Episodes', fontsize=40)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower left', fontsize=25)
        plt.tight_layout()
        plt.show()

        
    def plot_actions(self):
        sns.set_style("whitegrid")

        mutation_episode = self.training_params[kc.HUMAN_LEARNING_LENGTH]

        if self.possible_agents:
            random_agents = random.sample(self.possible_agents, 1)
            human_agents = [agent for agent in self.human_agents if agent.origin == self.origin[0] and agent.destination == self.destination[0]]
            human_agents = random.sample(human_agents, 2)
        else:
            human_agents = random.sample(self.human_agents, 2)

        plt.figure(figsize=(20, 12), dpi=200)

        colors = ['tab:red', 'tab:blue', 'tab:green']
        for idx, human in enumerate(human_agents):
            color = colors[idx]
            y_values = self.action_table_humans[human.id]
            x_values = np.arange(len(y_values))
            # Add jitter to the x-axis values
            jittered_x = x_values + np.random.uniform(-0.1, 0.1, size=len(x_values))
            plt.scatter(jittered_x, y_values, color=color, s=50, label=f'Human Agent {human.id} with o-d pair: {human.origin} - {human.destination}')

        if self.possible_agents:
            for agent_index in random_agents:
                if self.possible_agents:
                    for idx, human in enumerate(human_agents):
                        color = colors[idx]
                        x_values = [mutation_episode, mutation_episode + 1]
                        y_values = [self.action_table_humans[human.id][-1], self.action_table_humans[human.id][0]]
                        plt.plot(x_values, y_values, linestyle='-', color=color, linewidth=3)
                        y_values = self.action_table_humans[human.id]
                        x_values = np.arange(len(y_values))
                        jittered_x = x_values + np.random.uniform(-0.1, 0.1, size=len(x_values))
                        plt.scatter(jittered_x, y_values, color=color, s=50)
                    y_values = self.action_table[agent_index]
                    x_values = np.arange(mutation_episode + 1, mutation_episode + len(y_values) + 1)
                    jittered_x = x_values + np.random.uniform(-0.1, 0.1, size=len(x_values))
                    plt.scatter(jittered_x, y_values, color='tab:green', s=50, label=f'Machine Agent with o-d pair: {self.origin[0]} - {self.destination[0]}')
                else:
                    y_values = self.action_table_humans[int(agent_index)]
                    x_values = np.arange(len(y_values))
                    jittered_x = x_values + np.random.uniform(-0.1, 0.1, size=len(x_values))
                    plt.scatter(jittered_x, y_values, color='tab:green', s=50)

            plt.axvline(x=mutation_episode, color='k', linestyle='--', label='Mutation Time', linewidth=3)

        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.xlabel('Episode', fontsize=40)
        plt.ylabel('Action', fontsize=40)
        #plt.title('Actions Over Episodes', fontsize=40)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower left', fontsize=30)
        plt.tight_layout()
        plt.yticks([0, 1, 2])  # Set y-axis ticks to only 0, 1, and 2
        plt.show()


    def empty_reward_action_tables(self):

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


    def calculate_rewards(self, sumo_df):
        average_reward = -1 * sumo_df['travel_time'].mean() #/ self.overall_min_travel_time

        if average_reward > self.min_reward:
            self.min_reward = average_reward

        return average_reward


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=self.agent_params[kc.NUM_HUMAN_AGENTS], shape=(self.simulation_params[kc.NUMBER_OF_PATHS],), dtype=np.float32)


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.simulation_params[kc.NUMBER_OF_PATHS])
