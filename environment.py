import pandas as pd
import random

from keychain import Keychain as kc
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

    def step(self, joint_action):   # For now, returns random rewards
        agent_ids = joint_action[kc.AGENT_ID]
        rewards = [-1 * random.uniform(20, 50) for i in range(len(agent_ids))]
        joint_reward = pd.DataFrame({kc.AGENT_ID : agent_ids, kc.REWARD : rewards})
        return joint_reward, None, True

    def calculate_rewards():
        pass