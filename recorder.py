import matplotlib.pyplot as plt

from keychain import Keychain as kc

class Recorder:

    def __init__(self, params):
        self.remember_every = params[kc.REMEMBER_EVERY]
        self.mode = params[kc.RECORDER_MODE]

#################### REMEMBER FUNCTIONS
    
    def remember_all(self, joint_action, joint_reward, agents):
        self.remember_actions(joint_action)
        self.remember_rewards(joint_reward)
        self.remember_agents_status(agents)

    def remember_actions(self, joint_action):
        pass


    def remember_rewards(self, joint_reward):
        pass


    def remember_agents_status(self, agents):
        pass

########################################
    
#################### SAVING PLOTS & VISUALIZATION