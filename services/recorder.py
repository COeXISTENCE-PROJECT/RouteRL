import os
import pandas as pd

from keychain import Keychain as kc
from utilities import list_to_string
from utilities import make_dir
from bs4 import BeautifulSoup


class Recorder:

    """
    Class to record the training process.
    """

    def __init__(self, params):

        self.episodes_folder = make_dir([kc.RECORDS_FOLDER, kc.EPISODES_LOGS_FOLDER])
        self.agents_folder = make_dir([kc.RECORDS_FOLDER, kc.AGENTS_LOGS_FOLDER])
        self.detector_folder = make_dir([kc.RECORDS_FOLDER, kc.DETECTOR_LOGS_FOLDER])
        self.sim_length_file_path = self.get_sim_length_file_path()

        self.saved_episodes = list()

        print(f"[SUCCESS] Recorder is now here to record!")


#################### INIT HELPERS
    
    def get_sim_length_file_path(self):
        log_file_path = make_dir(kc.RECORDS_FOLDER, kc.SIMULATION_LENGTH_LOG_FILE_NAME)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        return log_file_path
    
####################
        

#################### REMEMBER FUNCTIONS
    
    def remember_all(self, episode, joint_action, joint_reward, agents, last_sim_duration, det_dict):
        self.saved_episodes.append(episode)
        self.remember_episode(episode, joint_action, joint_reward)
        self.remember_agents_status(episode, agents)
        self.remember_last_sim_duration(last_sim_duration)
        self.remember_detector(episode, det_dict)


    def remember_episode(self, episode, joint_action, joint_reward):
        origins, dests, actions = joint_action[kc.AGENT_ORIGIN], joint_action[kc.AGENT_DESTINATION], joint_action[kc.ACTION]
        joint_action[kc.SUMO_ACTION] = [f"{origins[i]}_{dests[i]}_{action}" for i, action in enumerate(actions)]
        merged_df = pd.merge(joint_action, joint_reward, on=kc.AGENT_ID)
        joint_df = pd.DataFrame(joint_action)
        joint_df = joint_df.drop(columns=['kind'])
        merged_df.to_csv(make_dir(self.episodes_folder, f"ep_ep{episode}.csv"), index = False)
        #joint_df.to_csv(make_dir(self.episodes_folder, f"ep_ep{episode}.csv"), index = False)

    def remember_detector(self,episode, det_dict):
            df = pd.DataFrame(list(det_dict.items()), columns=['detid', 'flow'])
            df.to_csv(make_dir(self.detector_folder, f'detector_ep{episode}.csv'),index=False)




    def remember_agents_status(self, episode, agents):
        #agents_df_cols = [kc.AGENT_ID, kc.AGENT_KIND, kc.COST_TABLE, kc.TO_MUTATE, kc.Utility, kc.ALPHA, kc.BETA, kc.EPSILON, kc.EPSILON_DECAY_RATE, kc.GAMMA, kc.Q_TABLE]
        agents_df_cols = [kc.AGENT_ID, kc.COST_TABLE, kc.Utility, kc.Noise]
        agents_df = pd.DataFrame(columns = agents_df_cols)
        for agent in agents:
            id, kind = agent.id, agent.kind
            beta, alpha, cost, q_table, epsilon, epsilon_decay_rate, gamma, to_mutate, utility, noise = [kc.NOT_AVAILABLE] * 10
            cost, utility, noise = [kc.NOT_AVAILABLE] * 3
            if kind == kc.TYPE_HUMAN:
                beta, alpha, cost, to_mutate, utility, noise = agent.beta, agent.alpha, list_to_string(agent.cost, ' , '), (agent.mutate_to != None), list_to_string(agent.utility), agent.noise
            elif kind == kc.TYPE_MACHINE:
                alpha, epsilon, epsilon_decay_rate, gamma, q_table = agent.alpha, agent.epsilon, agent.epsilon_decay_rate, agent.gamma, list_to_string(agent.q_table, ' , ')
            #row_data = [id, kind, cost, to_mutate, utility, alpha, beta, epsilon, epsilon_decay_rate, gamma, q_table]
            row_data = [id, cost, utility, noise]
            agents_df.loc[len(agents_df.index)] = {key : value for key, value in zip(agents_df_cols, row_data)}
        agents_df.to_csv(make_dir(self.agents_folder, f"ep{episode}.csv"), index = False)


    def remember_last_sim_duration(self, last_sim_duration):
        with open(self.sim_length_file_path, "a") as file:
            file.write(f"{last_sim_duration}\n")

####################
            
    
        