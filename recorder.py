import matplotlib.pyplot as plt
import random

from keychain import Keychain as kc

class Recorder:

    def __init__(self, agents, params):
        self.agents = agents

        self.remember_every = params[kc.REMEMBER_EVERY]
        self.mode = params[kc.RECORDER_MODE]
        self.track_human, self.track_machine = self.get_tracking_agents(params[kc.TRACK_HUMAN], params[kc.TRACK_MACHINE])

    
    def get_tracking_agents(self, if_track_human, if_track_machine):
        track_human, track_machine = None, None
        if if_track_human:
            track_human = random.choice(self.agents)
            while track_human.kind != kc.TYPE_HUMAN:
                track_human = random.choice(self.agents)
        if if_track_machine:
            track_machine = random.choice(self.agents)
            while track_machine.kind != kc.TYPE_MACHINE:
                track_machine = random.choice(self.agents) 
        return track_human, track_machine


#################### REMEMBER FUNCTIONS
    
    def remember_all(self, joint_action, joint_reward, agents):
        self.remember_actions(joint_action)
        self.remember_rewards(joint_reward)
        self.remember_agents_status(agents)
        self.remember_human(joint_action, joint_reward, agents)
        self.remember_machine(joint_action, joint_reward, agents)

    def remember_actions(self, joint_action):
        pass


    def remember_rewards(self, joint_reward):
        pass


    def remember_agents_status(self, agents):
        pass


    def remember_human(self, joint_action, joint_reward, agents):
        if self.track_human:
            pass
        


    def remember_machine(self, joint_action, joint_reward, agents):
        if self.track_machine:
            pass

########################################
    
#################### SAVING PLOTS & VISUALIZATION