import matplotlib.pyplot as plt
import os
import pandas as pd
import random

from collections import Counter

from keychain import Keychain as kc
from services.utils import make_dir, list_to_string, string_to_list

class Recorder:

    def __init__(self, agents, params):
        self.agents, self.humans, self.machines = list(), list(), list()
        self.update_agents(agents)

        self.remember_every = params[kc.REMEMBER_EVERY]
        self.save, self.show = self.resolve_mode(params[kc.RECORDER_MODE])
        self.human_to_track, self.machine_to_track = self.get_tracking_agents(params[kc.TRACK_HUMAN], params[kc.TRACK_MACHINE])

        self.episodes_folder = make_dir([kc.RECORDS_FOLDER, kc.EPISODES_LOGS_FOLDER])
        self.machines_folder = make_dir([kc.RECORDS_FOLDER, kc.MACHINES_LOG_FOLDER])
        self.humans_folder = make_dir([kc.RECORDS_FOLDER, kc.HUMANS_LOG_FOLDER])

        self.episodes = list()


#################### INIT HELPERS
        
    def update_agents(self, agents):
        self.agents, self.humans, self.machines = agents, list(), list()
        for agent in agents:
            if agent.kind == kc.TYPE_HUMAN:
                self.humans.append(agent.id)
            elif agent.kind == kc.TYPE_MACHINE:
                self.machines.append(agent.id)

    
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


    def resolve_mode(self, mode):
        if mode == kc.PLOT_AND_SAVE:
            return True, True
        elif mode == kc.PLOT_ONLY:
            return False, True
        elif mode == kc.SAVE_ONLY:
            return True, False
        
########################################
        

#################### REMEMBER FUNCTIONS
    
    def remember_all(self, episode, joint_action, joint_reward, agents):
        if not (episode % self.remember_every):
            self.episodes.append(episode)
            self.update_agents(agents)
            self.remember_episode(episode, joint_action, joint_reward)
            self.remember_agents_status(episode)


    def remember_episode(self, episode, joint_action, joint_reward):
        origins, dests, actions = joint_action[kc.AGENT_ORIGIN], joint_action[kc.AGENT_DESTINATION], joint_action[kc.ACTION]
        joint_action[kc.SUMO_ACTION] = [f"{origins[i]}_{dests[i]}_{action}" for i, action in enumerate(actions)]
        merged_df = pd.merge(joint_action, joint_reward, on=kc.AGENT_ID)
        merged_df.to_csv(make_dir(self.episodes_folder, f"ep{episode}.csv"), index = False)


    def remember_agents_status(self, episode):
        self.remember_machines_status(episode)
        self.remember_humans_status(episode)


    def remember_machines_status(self, episode):
        machines_df_cols = [kc.AGENT_ID, kc.ALPHA, kc.EPSILON, kc.EPSILON_DECAY_RATE, kc.GAMMA, kc.Q_TABLE]
        machines_df = pd.DataFrame(columns = machines_df_cols)
        machine_agents = self.get_agents_from_ids(self.machines)
        for machine in machine_agents:
            row_data = [machine.id, machine.alpha, machine.epsilon, machine.epsilon_decay_rate, machine.gamma, 
                        list_to_string(machine.q_table, ' , ')]
            machines_df.loc[len(machines_df.index)] = {key : value for key, value in zip(machines_df_cols, row_data)} 
        machines_df.to_csv(make_dir([self.machines_folder], f"machines_ep{episode}.csv"), index = False)
        

    def remember_humans_status(self, episode):
        humans_df_cols = [kc.AGENT_ID, kc.COST_TABLE]
        humans_df = pd.DataFrame(columns = humans_df_cols)
        human_agents = self.get_agents_from_ids(self.humans)
        for human in human_agents:
            row_data = [human.id, list_to_string(human.cost, ' , ')]
            humans_df.loc[len(humans_df.index)] = {key : value for key, value in zip(humans_df_cols, row_data)}  
        humans_df.to_csv(make_dir([self.humans_folder], f"humans_ep{episode}.csv"), index = False)

########################################
        
#################### REWIND
        
    def rewind(self):
        self.visualize_mean_rewards()
        self.visualize_flows()
        self.visualize_tracking_agent_data()


########################################
    
#################### MEAN REWARDS
        
    def visualize_mean_rewards(self):
        mean_rewards = self.retrieve_mean_rewards()

        plt.plot(self.episodes, mean_rewards[kc.HUMANS], label="Humans")
        plt.plot(self.episodes, mean_rewards[kc.MACHINES], label="Machines")
        plt.plot(self.episodes, mean_rewards[kc.ALL], label="All")
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        plt.title('Mean Rewards Over Episodes')
        plt.legend()
        if self.save: plt.savefig(make_dir(kc.PLOTS_LOG_PATH, kc.REWARDS_PLOT_FILE_NAME))
        if self.show: plt.show()
        plt.close()

    
    def retrieve_mean_rewards(self):
        all_mean_rewards, mean_human_rewards, mean_machine_rewards = list(), list(), list()

        for episode in self.episodes:
            data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            data = pd.read_csv(data_path)
            ids, rewards = data[kc.AGENT_ID], data[kc.REWARD]
            human_rewards, machine_rewards = list(), list()
            for id, reward in zip(ids, rewards):
                if id in self.humans:
                    human_rewards.append(reward)
                else:
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
    
    
#################### TRACKING ONE AGENT
    

    def visualize_tracking_agent_data(self):
        if not (self.human_to_track is None):
            self.visualize_tracking_human_data()
        if not (self.machine_to_track is None):
            self.visualize_tracking_machine_data()


########################################
    

#################### TRACKING ONE HUMAN
    
    def retrieve_tracking_human_data(self):
        collected_rewards, picked_actions, current_costs = list(), list(), list()
        for episode in self.episodes:
            experience_data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            experience_data = pd.read_csv(experience_data_path)

            humans_data_path = os.path.join(self.humans_folder, f"humans_ep{episode}.csv")
            humans_data = pd.read_csv(humans_data_path)

            merged_df = pd.merge(experience_data, humans_data, on=kc.AGENT_ID, how="left")

            for _, row in merged_df.iterrows():
                if row[kc.AGENT_ID] == self.human_to_track.id:
                    collected_rewards.append(row[kc.REWARD])
                    picked_actions.append(row[kc.ACTION])
                    current_costs.append(row[kc.COST_TABLE])

        experiences_df = pd.DataFrame({
            kc.REWARD : collected_rewards,
            kc.ACTION : picked_actions,
            "costs" : current_costs
        })

        return experiences_df
    

    def visualize_tracking_human_data(self):
        tracking_human_data = self.retrieve_tracking_human_data()
        self.introduce_agent(self.human_to_track)

        costs = tracking_human_data["costs"]
        actions = tracking_human_data[kc.ACTION]
        rewards = tracking_human_data[kc.REWARD]

        parsed_costs = list()
        for cost in costs:
            cost = string_to_list(cost, " , ")
            parsed_costs.append([float(value) for value in cost])

        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        # Plot on each subplot
        axs[0].plot(self.episodes, [cost[0] for cost in parsed_costs], label="Action 0")
        axs[0].plot(self.episodes, [cost[1] for cost in parsed_costs], label="Action 1")
        axs[0].plot(self.episodes, [cost[2] for cost in parsed_costs], label="Action 2")
        axs[0].legend()
        axs[0].set_title('Cost Table Over Episodes')

        axs[1].plot(self.episodes, rewards)
        axs[1].set_title('Collected Rewards Over Episodes')

        axs[2].step(self.episodes, actions, where='mid', linestyle='None', marker='o')
        axs[2].set_title('Picked Actions Over Episodes')

        fig.suptitle('One Human Experience')
        plt.tight_layout()

        if self.save: plt.savefig(make_dir(kc.PLOTS_LOG_PATH, kc.ONE_HUMAN_PLOT_FILE_NAME))
        if self.show: plt.show()
        plt.close()
    

########################################
        

#################### TRACKING ONE MACHINE

    def retrieve_tracking_machine_data(self):
        collected_rewards, picked_actions = list(), list()
        current_q_tables, current_epsilons = list(), list()
        for episode in self.episodes:
            experience_data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            experience_data = pd.read_csv(experience_data_path)

            machines_data_path = os.path.join(self.machines_folder, f"machines_ep{episode}.csv")
            machines_data = pd.read_csv(machines_data_path)

            merged_df = pd.merge(experience_data, machines_data, on=kc.AGENT_ID, how="left")

            for _, row in merged_df.iterrows():
                if row[kc.AGENT_ID] == self.machine_to_track.id:
                    collected_rewards.append(row[kc.REWARD])
                    picked_actions.append(row[kc.ACTION])
                    current_q_tables.append(row[kc.Q_TABLE])
                    current_epsilons.append(row[kc.EPSILON])

        experiences_df = pd.DataFrame({
            kc.REWARD : collected_rewards,
            kc.ACTION : picked_actions,
            kc.Q_TABLE : current_q_tables,
            "epsilons" : current_epsilons
        })

        return experiences_df
    

    def visualize_tracking_machine_data(self):
        tracking_machine_data = self.retrieve_tracking_machine_data()
        self.introduce_agent(self.machine_to_track)

        q_tables = tracking_machine_data[kc.Q_TABLE]
        actions = tracking_machine_data[kc.ACTION]
        rewards = tracking_machine_data[kc.REWARD]
        epsilons = tracking_machine_data["epsilons"]

        parsed_q = list()
        for q in q_tables:
            q = string_to_list(q, " , ")
            parsed_q.append([float(value) for value in q])

        fig, axs = plt.subplots(4, 1, figsize=(8, 12))

        # Plot on each subplot
        axs[0].plot(self.episodes, [q[0] for q in parsed_q], label="Action 0")
        axs[0].plot(self.episodes, [q[1] for q in parsed_q], label="Action 1")
        axs[0].plot(self.episodes, [q[2] for q in parsed_q], label="Action 2")
        axs[0].legend()
        axs[0].set_title('Q-Table Over Episodes')

        axs[1].plot(self.episodes, rewards)
        axs[1].set_title('Collected Rewards Over Episodes')

        axs[2].step(self.episodes, actions, where='mid', linestyle='None', marker='o')
        axs[2].set_title('Picked Actions Over Episodes')

        axs[3].plot(self.episodes, epsilons)
        axs[3].set_title('Epsilon Over Episodes')

        fig.suptitle('One Machine Experience')
        plt.tight_layout()

        if self.save: plt.savefig(make_dir(kc.PLOTS_LOG_PATH, kc.ONE_MACHINE_PLOT_FILE_NAME))
        if self.show: plt.show()
        plt.close()
    

########################################
        
#################### FLOWS
    
    def visualize_flows(self):

        ever_picked, all_selections = self.retrieve_flows()

        for pick in ever_picked:
            plt.plot(self.episodes, [selections[pick] for selections in all_selections], label=pick)

        plt.xlabel('Episodes')
        plt.ylabel('Population')
        plt.title('Population in Routes Over Episodes')
        plt.legend()
        if self.save: plt.savefig(make_dir(kc.PLOTS_LOG_PATH, kc.FLOWS_PLOT_FILE_NAME))
        if self.show: plt.show()
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

#################### HELPERS
        
    def get_agent_from_id(self, id):
        for agent in self.agents:
            if agent.id == id:
                return agent
        return None
    

    def get_agents_from_ids(self, ids):
        found = list()
        for agent in self.agents:
            if agent.id in ids:
                found.append(agent)
        return found


    def mean(self, data):
        return sum(data) / len(data)
    

    def introduce_agent(self, agent):
        print(f"[INFO] Now visualizing agent #{agent.id} from kind {agent.kind} with OD: {agent.origin}-{agent.destination} at start time {agent.start_time}.")
    

