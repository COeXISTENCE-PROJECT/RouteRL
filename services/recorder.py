import os
import pandas as pd

from keychain import Keychain as kc
from utilities import list_to_string
from utilities import make_dir


class Recorder:

    """
    Class to record the training process.
    """

    def __init__(self, params):

        self.episodes_folder = make_dir([kc.RECORDS_FOLDER, kc.EPISODES_LOGS_FOLDER])
        self.agents_folder = make_dir([kc.RECORDS_FOLDER, kc.AGENTS_LOGS_FOLDER])
        self.sim_length_file_path = self.get_txt_file_path(kc.SIMULATION_LENGTH_LOG_FILE_NAME)
        self.loss_file_path = self.get_txt_file_path(kc.LOSSES_LOG_FILE_NAME)

        self.saved_episodes = list()

        print(f"[SUCCESS] Recorder is now here to record!")


#################### INIT HELPERS
    
    def get_txt_file_path(self, filename):
        log_file_path = make_dir(kc.RECORDS_FOLDER, filename)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        return log_file_path
    
####################
        

#################### REMEMBER FUNCTIONS
    
    def remember_all(self, episode, joint_action, joint_observation, agents, last_sim_duration):
        self.saved_episodes.append(episode)
        self.remember_episode(episode, joint_action, joint_observation)
        #self.remember_agents_status(episode, agents)
        self.remember_last_sim_duration(last_sim_duration)


    def remember_episode(self, episode, joint_action, joint_observation):
        origins, dests, actions = joint_action[kc.AGENT_ORIGIN], joint_action[kc.AGENT_DESTINATION], joint_action[kc.ACTION]
        joint_action[kc.SUMO_ACTION] = [f"{origins[i]}_{dests[i]}_{action}" for i, action in enumerate(actions)]
        merged_df = pd.merge(joint_action, joint_observation, on=kc.AGENT_ID)
        merged_df.to_csv(make_dir(self.episodes_folder, f"ep{episode}.csv"), index = False)


    def remember_agents_status(self, episode, agents):
        agents_df_cols = [kc.AGENT_ID, kc.AGENT_KIND, kc.COST_TABLE, kc.TO_MUTATE, kc.ALPHA, kc.BETA, kc.EPSILON, kc.EPSILON_DECAY_RATE, kc.GAMMA, kc.Q_TABLE]
        agents_df = pd.DataFrame(columns = agents_df_cols)
        for agent in agents:
            id, kind = agent.id, agent.kind
            beta, alpha, cost, q_table, epsilon, epsilon_decay_rate, gamma, to_mutate = [kc.NOT_AVAILABLE] * 8
            if kind == kc.TYPE_HUMAN:
                beta, alpha, cost, to_mutate = agent.beta, agent.alpha, list_to_string(agent.cost, ' , '), (agent.mutate_to != None)
            elif (kind == kc.TYPE_MACHINE) or (kind == kc.TYPE_MACHINE_2):
                alpha, epsilon, epsilon_decay_rate, gamma, q_table = agent.alpha, agent.epsilon, agent.epsilon_decay_rate, agent.gamma, list_to_string(agent.q_table, ' , ')
            row_data = [id, kind, cost, to_mutate, alpha, beta, epsilon, epsilon_decay_rate, gamma, q_table]
            agents_df.loc[len(agents_df.index)] = {key : value for key, value in zip(agents_df_cols, row_data)}
        agents_df.to_csv(make_dir(self.agents_folder, f"ep{episode}.csv"), index = False)


    def remember_last_sim_duration(self, last_sim_duration):
        with open(self.sim_length_file_path, "a") as file:
            file.write(f"{last_sim_duration}\n")

    def save_losses(self, agents):
        losses = list()
        for a in agents:
            loss = getattr(a.model, 'loss', None)
            if loss is not None:
                losses.append(loss)
        if losses:
            mean_losses = [0] * len(losses[-1])
            for loss in losses:
                for i, l in enumerate(loss):
                    mean_losses[i] += l
            mean_losses = [m / len(losses) for m in mean_losses]
            with open(self.loss_file_path, "w") as file:
                for m_l in mean_losses:
                    file.write(f"{m_l}\n")

####################
            
    
        