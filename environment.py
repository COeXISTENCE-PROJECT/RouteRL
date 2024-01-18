import pandas as pd
import random

from simulator import Simulator


class TrafficEnvironment:

    """
    To be implemented
    """

    def __init__(self): # get params for simulator
        simulator = Simulator() # pass params for simulator
        # Initialize network
        
        # Create demand
        # Create paths
        # Calculate free flows
        return None

    def reset(self):
        return None

    def step(self, joint_action):
        agent_ids = joint_action['id']
        rewards = [-1 * random.uniform(20, 50) for i in range(len(agent_ids))]
        joint_reward = pd.DataFrame({'id' : agent_ids, 'reward' : rewards})
        return joint_reward, None, True

    def calculate_rewards():
        pass