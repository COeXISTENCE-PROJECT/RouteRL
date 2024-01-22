import pandas as pd
import random
from prettytable import PrettyTable

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
        simulator = Simulator() # pass params for simulator

        #### 
        # Create 600 agents
        self.print_agents(agents, print_every=50)
        
        
        return None

    def reset(self):
        return None

    def step(self, joint_action):   # For now, returns random rewards
        agent_ids = joint_action[kc.AGENT_ID]
        rewards = [-1 * random.uniform(20, 50) for i in range(len(agent_ids))]
        joint_reward = pd.DataFrame({kc.AGENT_ID : agent_ids, kc.REWARD : rewards})
        return joint_reward, None, True

    def calculate_rewards(self):
        pass

    def print_agents(self, agents, print_every=1): # Should this be even here?
        table = PrettyTable()
        table.field_names = kc.AGENT_ATTRIBUTES

        for a in agents:
            if not (a.id % print_every):
                table.add_row([a.id, a.origin, a.destination, a.start_time, a.__class__.__name__])

        if print_every > 1: print("--- Showing every %d agent ---" % (print_every))
        print(table)
