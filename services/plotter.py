import matplotlib.pyplot as plt
import os
import pandas as pd
import random

from collections import Counter

from keychain import Keychain as kc
from utilities import make_dir
from utilities import string_to_list


class Plotter:

    """
    This class is to plot the results of the training
    """
    def __init__(self, episodes, mutation_time, params):

        self.episodes_folder = make_dir([kc.RECORDS_FOLDER, kc.EPISODES_LOGS_FOLDER])
        self.agents_folder = make_dir([kc.RECORDS_FOLDER, kc.AGENTS_LOGS_FOLDER])
        self.sim_length_file_path = make_dir([kc.RECORDS_FOLDER, kc.SIMULATION_LOG_FOLDER], kc.SIMULATION_LENGTH_LOG_FILE_NAME)

        self.episodes = episodes
        self.mutation_time = mutation_time

        print(f"[SUCCESS] Plotter is now here to plot!")


    def visualize_all(self):
        self.visualize_mean_rewards()
        self.visualize_flows()
        #self.visualize_tracking_agent_data()
        self.visualize_sim_length()


#################### MEAN REWARDS
        
    def visualize_mean_rewards(self):
        mean_rewards = self.retrieve_mean_rewards()

        plt.plot(self.episodes, mean_rewards[kc.HUMANS], label="Humans")
        machine_episodes = [ep for idx, ep in enumerate(self.episodes) if (mean_rewards[kc.MACHINES][idx] is not None)]
        machine_rewards = [reward for reward in mean_rewards[kc.MACHINES] if (reward is not None)]
        plt.plot(machine_episodes, machine_rewards, label="Machines")
        plt.plot(self.episodes, mean_rewards[kc.ALL], label="All")
        plt.axvline(x = self.mutation_time, label = 'Mutation Time', color = 'r', linestyle = '--')
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        plt.title('Mean Rewards Over Episodes')
        plt.legend()
        plt.savefig(make_dir(kc.PLOTS_FOLDER, kc.REWARDS_PLOT_FILE_NAME))
        #plt.show()
        plt.close()

    
    def retrieve_mean_rewards(self):
        all_mean_rewards, mean_human_rewards, mean_machine_rewards = list(), list(), list()

        for episode in self.episodes:
            data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            data = pd.read_csv(data_path)
            kinds, rewards = data[kc.AGENT_KIND], data[kc.REWARD]
            human_rewards, machine_rewards = list(), list()
            for kind, reward in zip(kinds, rewards):
                if kind == kc.TYPE_HUMAN:
                    human_rewards.append(reward)
                elif kind == kc.TYPE_MACHINE:
                    machine_rewards.append(reward)
                
            mean_human_rewards.append(self.mean(human_rewards))
            mean_machine_rewards.append(self.mean(machine_rewards))
            all_mean_rewards.append(self.mean(rewards))

        mean_rewards = pd.DataFrame({
            kc.HUMANS: mean_human_rewards,
            kc.MACHINES: mean_machine_rewards,
            kc.ALL: all_mean_rewards
        })

        return mean_rewards


########################################
    
#################### FLOWS
    
    def visualize_flows(self):

        ever_picked, all_selections = self.retrieve_flows()
        ever_picked = list(ever_picked)
        ever_picked.sort()
        
        for pick in ever_picked:
            plt.plot(self.episodes, [selections[pick] for selections in all_selections], label=pick)

        plt.xlabel('Episodes')
        plt.ylabel('Population')
        plt.title('Population in Routes Over Episodes')
        plt.legend()
        plt.savefig(make_dir(kc.PLOTS_FOLDER, kc.FLOWS_PLOT_FILE_NAME))
        #plt.show()
        plt.close()


    def retrieve_flows(self):

        all_selections = list()
        ever_picked = set()

        for episode in self.episodes:
            experience_data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            experience_data = pd.read_csv(experience_data_path)
            actions = experience_data[kc.SUMO_ACTION]

            ever_picked.update(actions)
            actions_counter = Counter(actions)
            all_selections.append(actions_counter)

        return ever_picked, all_selections
        
########################################
    

#################### SIM LENGTH
    
    def visualize_sim_length(self):
        sim_lengths = self.retrieve_sim_length()
        plt.plot(self.episodes, sim_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Simulation Length')
        plt.title('Simulation Length Over Episodes')
        plt.savefig(make_dir(kc.PLOTS_FOLDER, kc.SIMULATION_LENGTH_PLOT_FILE_NAME))
        #plt.show()
        plt.close()

    def retrieve_sim_length(self):
        sim_lengths = list()
        with open(self.sim_length_file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                sim_lengths.append(int(line.strip()))
        return sim_lengths
    
########################################
    
    def mean(self, data):
        if len(data) == 0:
            return None
        return sum(data) / len(data)