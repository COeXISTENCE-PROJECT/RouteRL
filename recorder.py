import matplotlib.pyplot as plt
import pandas as pd
import random

from keychain import Keychain as kc
from services.utils import make_dir, list_to_string

class Recorder:

    def __init__(self, agents, params):
        self.agents, self.humans, self.machines = list(), list(), list()
        self.update_agents(agents)

        self.remember_every = params[kc.REMEMBER_EVERY]
        self.mode = params[kc.RECORDER_MODE]
        self.human_to_track, self.machine_to_track = self.get_tracking_agents(params[kc.TRACK_HUMAN], params[kc.TRACK_MACHINE])


    def update_agents(self, agents):
        self.agents, self.humans, self.machines = agents, list(), list()
        for agent in agents:
            if agent.kind == kc.TYPE_HUMAN:
                self.humans.append(agent)
            elif agent.kind == kc.TYPE_MACHINE:
                self.machines.append(agent)

    
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
    
    def remember_all(self, episode, joint_action, joint_reward, agents):
        if not (episode % self.remember_every):
            self.update_agents(agents)
            self.remember_actions(episode, joint_action)
            self.remember_rewards(episode, joint_reward)
            self.remember_agents_status(episode)
            self.track_human(episode, joint_action, joint_reward)
            self.track_machine(episode, joint_action, joint_reward)

    def remember_actions(self, episode, joint_action):
        joint_action.drop([kc.AGENT_ORIGIN, kc.AGENT_DESTINATION, kc.AGENT_START_TIME], axis=1, inplace=True)
        joint_action.to_csv(make_dir(kc.RECORDS_PATH, f"actions_ep{episode}.csv", [kc.ACTIONS_LOGS_PATH]), index = False)

 
    def remember_rewards(self, episode, joint_reward):
        joint_reward.to_csv(make_dir(kc.RECORDS_PATH, f"rewards_ep{episode}.csv", [kc.REWARDS_LOGS_PATH]), index = False)


    def remember_agents_status(self, episode):
        self.remember_machines_status(episode)
        self.remember_humans_status(episode)


    def remember_machines_status(self, episode):
        machines_df_cols = ["id", "alpha", "epsilon", "eps_decay", "gamma", "q_table"]
        machines_df = pd.DataFrame(columns = machines_df_cols)
        for machine in self.machines:
            row_data = [machine.id, machine.alpha, machine.epsilon, machine.epsilon_decay_rate, machine.gamma, list_to_string(machine.q_table)]
            machines_df.loc[len(machines_df.index)] = {key : value for key, value in zip(machines_df_cols, row_data)} 
        machines_df.to_csv(make_dir(kc.RECORDS_PATH, f"machines_ep{episode}.csv", [kc.MACHINES_LOG_PATH]), index = False)
        


    def remember_humans_status(self, episode):
        humans_df_cols = ["id", "cost"]
        humans_df = pd.DataFrame(columns = humans_df_cols)
        for human in self.humans:
            row_data = [human.id, list_to_string(human.cost)]
            humans_df.loc[len(humans_df.index)] = {key : value for key, value in zip(humans_df_cols, row_data)}  
        humans_df.to_csv(make_dir(kc.RECORDS_PATH, f"humans_ep{episode}.csv", [kc.HUMANS_LOG_PATH]), index = False)


    def track_human(self, episode, joint_action, joint_reward):
        if self.human_to_track:
            pass
        


    def track_machine(self, episode, joint_action, joint_reward):
        if self.machine_to_track:
            pass

########################################
    
#################### SAVING PLOTS & VISUALIZATION