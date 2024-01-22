import pandas as pd
import random
from prettytable import PrettyTable
import numpy as np

from keychain import Keychain as kc
from simulator import Simulator
from agent import Agent

class TrafficEnvironment:

    """
    To be implemented
    """

    def __init__(self, agents): # get params for simulator
        # Initialize network
        # Create demand
        # Create paths
        # Calculate free flows
        # done
        self.simulator = Simulator(agents)  # pass params for simulator, and only the number of agents
        self.print_agents(agents, print_every=50)   # delete maybe

        print("[SUCCESS] Environment initiated!")



    def calculate_free_flow_time(self):

        free_flow_cost = self.simulator.calculate_free_flow_time()

        return free_flow_cost
        
        

    def reset(self):
        return None



    def step(self, joint_action):   # For now, returns random rewards
        agent_ids = joint_action[kc.AGENT_ID]

        ####
        #### Feed agents actions to SUMO and get travel times
        ####

        #df1,df2 = self.simulator.run_simulation_iteration(joint_action, "agent_data.csv")#the last number of is the length of the simulation in seconds


        ###### random - will change
        sumo_df = pd.DataFrame({
            'id': agent_ids,
            'travel_times': np.random.uniform(low=1, high=100, size=len(agent_ids))
        })

        #### Calculate joint reward based on travel times returned by SUMO
        joint_reward = self.calculate_rewards(sumo_df)

        rewards = [joint_reward for i in range(len(agent_ids))]
        joint_reward = pd.DataFrame({kc.AGENT_ID : agent_ids, kc.REWARD : rewards})

        return joint_reward, None, True



    def calculate_rewards(self, sumo_df):
        average_travel_time = sumo_df['travel_times'].mean()

        return average_travel_time



    def print_agents(self, agents, print_every=1): # Should this be even here?
        table = PrettyTable()
        table.field_names = kc.AGENT_ATTRIBUTES

        for a in agents:
            if not (a.id % print_every):
                table.add_row([a.id, a.origin, a.destination, a.start_time, a.__class__.__name__])

        if print_every > 1: print("--- Showing every %dth agent ---" % (print_every))
        print(table)