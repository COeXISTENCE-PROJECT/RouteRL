import matplotlib.pyplot as plt
import os
import pandas as pd

from collections import Counter

from keychain import Keychain as kc
from utilities import make_dir



class Plotter:


    """
    This class is to plot the results of the training
    """


    def __init__(self, mutation_time, episodes_folder, agents_folder, sim_length_file_path, params):

        self.episodes_folder = episodes_folder
        self.agents_folder = agents_folder
        self.sim_length_file_path = sim_length_file_path
        self.free_flow_times_file_path = os.path.join(kc.RECORDS_FOLDER, kc.FREE_FLOW_TIMES_CSV_FILE_NAME)

        self.episodes = list()
        self.mutation_time = mutation_time

        print(f"[SUCCESS] Plotter is now here to plot!")



#################### VISUALIZE ALL

    def visualize_all(self, episodes):
        self.episodes = episodes

        self.visualize_free_flows()
        self.visualize_mean_rewards()
        self.visualize_flows()
        self.visualize_actions()
        self.visualize_action_shifts()
        self.visualize_sim_length()

####################
        


#################### FREE FLOWS
        
    def visualize_free_flows(self):
        save_to = make_dir(kc.PLOTS_FOLDER, kc.FF_TRAVEL_TIME_PLOT_FILE_NAME)

        free_flows = self.retrieve_free_flows()
        od_pairs = list(set(free_flows[['origins', 'destinations']].apply(lambda x: (x.iloc[0], x.iloc[1]), axis=1)))
        num_od_pairs = len(od_pairs)
        
        num_columns = 2 
        num_rows = (num_od_pairs + num_columns - 1) // num_columns

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, num_rows * 4))  # Adjust figsize as needed
        fig.tight_layout(pad=5.0)

        if num_rows > 1:   axes = axes.flatten()   # Flatten axes

        od_pairs.sort()
        for idx, od in enumerate(od_pairs):
            ax = axes[idx]
            subset = free_flows[free_flows[kc.ORIGINS] == od[0]]
            subset = subset[subset[kc.DESTINATIONS] == od[1]]
            for _, row in subset.iterrows():
                ax.bar(f"{row[kc.PATH_INDEX]}", row[kc.FREE_FLOW_TIME], label = f"Route: {int(row[kc.PATH_INDEX])}")

            ax.set_title(f"Free flow travel times in {od[0]}-{od[1]}")
            ax.set_xlabel('Route Index')
            ax.set_ylabel('Minutes')
            ax.legend()

        for ax in axes[idx+1:]:   ax.axis('off')    # Hide unused subplots if any

        #plt.show()
        plt.savefig(save_to)
        plt.close()
        print(f"[SUCCESS] Free-flow travel times are saved to {save_to}")



    def retrieve_free_flows(self):
        free_flows = pd.read_csv(self.free_flow_times_file_path)
        free_flows = free_flows.astype({kc.ORIGINS: 'int', kc.DESTINATIONS: 'int', kc.PATH_INDEX: 'int', kc.FREE_FLOW_TIME: 'float'})
        return free_flows
        

####################
        


#################### MEAN REWARDS
        
    def visualize_mean_rewards(self):
        save_to = make_dir(kc.PLOTS_FOLDER, kc.REWARDS_PLOT_FILE_NAME)

        mean_rewards = self.retrieve_mean_rewards()

        plt.figure(figsize=(12, 8))
        plt.plot(self.episodes, mean_rewards[kc.HUMANS], label="Humans")
        machine_episodes = [ep for ep in self.episodes if ep >= self.mutation_time]
        machine_rewards = [reward for idx, reward in enumerate(mean_rewards[kc.MACHINES]) if (self.episodes[idx] >= self.mutation_time)]
        plt.plot(machine_episodes, machine_rewards, label="Machines")
        plt.plot(self.episodes, mean_rewards[kc.ALL], label="All")

        plt.axvline(x = self.mutation_time, label = 'Mutation Time', color = 'r', linestyle = '--')
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        plt.title('Mean Rewards Over Episodes')
        plt.legend()

        #plt.show()
        plt.savefig(save_to)
        plt.close()
        print(f"[SUCCESS] Mean rewards are saved to {save_to}")


    
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

####################
    


#################### ACTIONS

    def visualize_actions(self):
        save_to = make_dir(kc.PLOTS_FOLDER, kc.ACTIONS_PLOT_FILE_NAME)

        all_actions, unique_actions = self.retrieve_actions()
        num_od_pairs = len(self.retrieve_all_od_pairs())
        
        # Determine the layout of the subplots (rows x columns)
        num_columns = 2 
        num_rows = (num_od_pairs + num_columns - 1) // num_columns  # Calculate rows needed
        
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, num_rows * 4))  # Adjust figsize as needed
        fig.tight_layout(pad=5.0)
        
        if num_rows > 1:   axes = axes.flatten()   # Flatten axes

        for idx, (od, actions) in enumerate(all_actions.items()):
            ax = axes[idx]
            for unique_action in unique_actions[od]:
                ax.plot(self.episodes, [ep_actions.get(unique_action, 0) for ep_actions in actions], label=f"{unique_action}")

            ax.axvline(x = self.mutation_time, label = 'Mutation Time', color = 'r', linestyle = '--') 
            ax.set_title(f"Actions for {od}")
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Population')
            ax.legend()

        # Hide unused subplots if any
        for ax in axes[idx+1:]:   ax.axis('off')    # Hide unused subplots if any

        plt.savefig(save_to)
        plt.close()
        print(f"[SUCCESS] Actions are saved to {save_to}")



    def retrieve_actions(self):
        all_od_pairs = self.retrieve_all_od_pairs()
        all_actions = {f"{od[0]} - {od[1]}" : list() for od in all_od_pairs}
        unique_actions = {f"{od[0]} - {od[1]}" : set() for od in all_od_pairs}
        for episode in self.episodes:
            episode_actions = {f"{od[0]} - {od[1]}" : list() for od in all_od_pairs}
            data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            episode_data = pd.read_csv(data_path)
            episode_origins, episode_destinations, actions = episode_data[kc.AGENT_ORIGIN], episode_data[kc.AGENT_DESTINATION], episode_data[kc.ACTION]
            for idx, action in enumerate(actions):
                episode_actions[f"{episode_origins[idx]} - {episode_destinations[idx]}"].append(action)
            for key in episode_actions.keys():
                unique_actions[key].update(episode_actions[key])
                episode_actions[key] = Counter(episode_actions[key])
            for key in all_actions.keys():
                all_actions[key].append(episode_actions[key])
        return all_actions, unique_actions
            
####################
    


#################### ACTION SHIFTS
    
    def visualize_action_shifts(self):
        save_to = make_dir(kc.PLOTS_FOLDER, kc.ACTIONS_SHIFTS_PLOT_FILE_NAME)

        all_od_pairs = self.retrieve_all_od_pairs()
        machine_episodes = [ep for ep in self.episodes if ep >= self.mutation_time]
        
        # Determine the layout of the subplots (rows x columns)
        num_columns = 2 
        num_rows = (len(all_od_pairs) + num_columns - 1) // num_columns  # Calculate rows needed
        
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, num_rows * 4))  # Adjust figsize as needed
        fig.tight_layout(pad=5.0)
        
        if num_rows > 1:   axes = axes.flatten()   # Flatten axes

        for idx, od in enumerate(all_od_pairs):
            ax = axes[idx]
            origin, destination = od

            all_actions, unique_actions = self.retrieve_selected_actions(origin, destination)

            for action in unique_actions:
                ax.plot(self.episodes, [ep_counter[action] / sum(ep_counter.values()) for ep_counter in all_actions[kc.TYPE_HUMAN]], label=f"Humans-{action}")

            for action in unique_actions: # Two loops just to make the plot legend alphabetical
                ax.plot(machine_episodes, [ep_counter[action] / sum(ep_counter.values()) for ep_counter in all_actions[kc.TYPE_MACHINE]], label=f"Machines-{action}")

            ax.axvline(x = self.mutation_time, label = 'Mutation Time', color = 'r', linestyle = '--')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Population')
            ax.set_title(f'Actions for {od} (Normalized by type population)')
            ax.legend()

        for ax in axes[idx+1:]:   ax.axis('off')    # Hide unused subplots if any

        plt.savefig(save_to)
        plt.close()
        print(f"[SUCCESS] Actions are saved to {save_to}")
        
    

    def retrieve_selected_actions(self, origin, destination):
        all_actions = {kc.TYPE_HUMAN: list(), kc.TYPE_MACHINE: list()}
        unique_actions = set()
        for episode in self.episodes:
            data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            data = pd.read_csv(data_path)
            data = data[(data[kc.AGENT_ORIGIN] == origin) & (data[kc.AGENT_DESTINATION] == destination)]
            kinds, actions = data[kc.AGENT_KIND], data[kc.ACTION]
            unique_actions.update(actions)
            actions_counters = {k : Counter() for k in set(kinds)}
            for kind, action in zip(kinds, actions):
                actions_counters[kind][action] += 1
            for kind in actions_counters.keys():
                all_actions[kind].append(actions_counters[kind])
        return all_actions, unique_actions
    
####################



#################### FLOWS
    
    def visualize_flows(self):
        save_to = make_dir(kc.PLOTS_FOLDER, kc.FLOWS_PLOT_FILE_NAME)

        ever_picked, all_selections = self.retrieve_flows()
        ever_picked = list(ever_picked)
        ever_picked.sort()
        
        plt.figure(figsize=(12, 8))

        for pick in ever_picked:
            plt.plot(self.episodes, [selections[pick] for selections in all_selections], label=pick)

        plt.axvline(x = self.mutation_time, label = 'Mutation Time', color = 'r', linestyle = '--')

        plt.xlabel('Episodes')
        plt.ylabel('Population')
        plt.title('Population in Routes Over Episodes')
        plt.legend()

        #plt.show()
        plt.savefig(save_to)
        plt.close()
        print(f"[SUCCESS] Flows are saved to {save_to}")



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
        
####################
    


#################### SIM LENGTH
    
    def visualize_sim_length(self):
        save_to = make_dir(kc.PLOTS_FOLDER, kc.SIMULATION_LENGTH_PLOT_FILE_NAME)

        sim_lengths = self.retrieve_sim_length()

        plt.figure(figsize=(12, 8))
        plt.plot(self.episodes, sim_lengths, label="Simulation timesteps")
        plt.axvline(x = self.mutation_time, label = 'Mutation Time', color = 'r', linestyle = '--')
        plt.xlabel('Episode')
        plt.ylabel('Simulation Length')
        plt.title('Simulation Length Over Episodes')
        plt.legend()
        
        #plt.show()
        plt.savefig(save_to)
        plt.close()
        print(f"[SUCCESS] Simulation lengths are saved to {save_to}")



    def retrieve_sim_length(self):
        sim_lengths = list()
        with open(self.sim_length_file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                sim_lengths.append(int(line.strip()))
        return sim_lengths
    
####################



#################### HELPERS

    def retrieve_all_od_pairs(self):
        all_od_pairs = list()
        data_path = os.path.join(self.episodes_folder, f"ep0.csv")
        episode_data = pd.read_csv(data_path)
        episode_data = episode_data[[kc.AGENT_ORIGIN, kc.AGENT_DESTINATION]]
        for _, row in episode_data.iterrows():
            origin, destination = row[kc.AGENT_ORIGIN], row[kc.AGENT_DESTINATION]
            all_od_pairs.append((origin, destination))
        all_od_pairs = list(set(all_od_pairs))
        return all_od_pairs
    


    def mean(self, data):
        if len(data) == 0:
            return None
        return sum(data) / len(data)
    
####################