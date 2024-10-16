import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from collections import Counter
from statistics import mean
from statistics import variance

from ..keychain import Keychain as kc
from ..utilities import make_dir
from ..utilities import running_average
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

class Plotter:

    """
    Plot the results of the training
    """

    def __init__(self, params):
        self.params = params
        self.phases = params[kc.PHASES]
        self.phase_names = params[kc.PHASE_NAMES]
        self.colors = params[kc.COLORS]
        self.phase_colors = list(reversed(self.colors))
        self.linestyles = params[kc.LINESTYLES]

        self.smooth_by = params[kc.SMOOTH_BY]
        self.default_width, self.default_height = params[kc.DEFAULT_WIDTH], params[kc.DEFAULT_HEIGHT]
        self.multimode_width, self.multimode_height = params[kc.MULTIMODE_WIDTH], params[kc.MULTIMODE_HEIGHT]
        self.default_num_columns = params[kc.DEFAULT_NUM_COLUMNS]
        self.records_folder = params[kc.RECORDS_FOLDER]

        make_dir(self.records_folder)
        self.episodes_folder = make_dir([self.records_folder, params[kc.EPISODES_LOGS_FOLDER]])
        self.sim_length_file_path = make_dir(self.records_folder, params[kc.SIMULATION_LENGTH_LOG_FILE_NAME])
        self.loss_file_path = make_dir(self.records_folder, params[kc.LOSSES_LOG_FILE_NAME])
        self.free_flow_times_file_path = os.path.join(self.records_folder, params[kc.FREE_FLOW_TIMES_CSV_FILE_NAME])

        self.saved_episodes = list()


#################### VISUALIZE ALL

    def plot(self):
        self.saved_episodes = self._get_episodes()

        self.visualize_mean_rewards()
        self.visualize_mean_travel_times()
        self.visualize_tt_distributions()
        self.visualize_actions()
        self.visualize_action_shifts()
        self.visualize_sim_length()
        self.visualize_losses()


    def _get_episodes(self):
        eps = list()
        if os.path.exists(self.episodes_folder):
            for file in os.listdir(self.episodes_folder):
                episode = int(file.split('ep')[1].split('.csv')[0])
                eps.append(episode)
        else:
            raise FileNotFoundError(f"Episodes folder does not exist!")
        return sorted(eps)

####################

#################### REWARDS
    
    def visualize_mean_rewards(self):
        save_to = make_dir(self.params[kc.PLOTS_FOLDER], self.params[kc.REWARDS_PLOT_FILE_NAME])
        all_mean_rewards = self._retrieve_data_per_kind(kc.REWARD, transform='mean')

        plt.figure(figsize=(self.default_width, self.default_height))

        for idx, (kind, ep_reward_dict) in enumerate(all_mean_rewards.items()):
            episodes = list(ep_reward_dict.keys())
            rewards = list(ep_reward_dict.values())
            smoothed_rewards = running_average(rewards, last_n=self.smooth_by)
            plt.plot(episodes, smoothed_rewards, color=self.colors[idx], label=kind)

        for phase_idx, phase in enumerate(self.phases):
            color = self.phase_colors[phase_idx % len(self.phase_colors)]
            plt.axvline(x=phase, label=self.phase_names[phase_idx], linestyle='--', color=color)

        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        plt.grid(True, axis='y')
        plt.title('Mean Rewards Over Episodes')
        plt.legend()

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Rewards are saved to {save_to}")
        
####################   


#################### TRAVEL TIMES
    
    def visualize_mean_travel_times(self):
        save_to = make_dir(self.params[kc.PLOTS_FOLDER], self.params[kc.TRAVEL_TIMES_PLOT_FILE_NAME])
        all_mean_tt = self._retrieve_data_per_kind(kc.TRAVEL_TIME, transform='mean')

        plt.figure(figsize=(self.default_width, self.default_height))

        for idx, (kind, ep_tt_dict) in enumerate(all_mean_tt.items()):
            episodes = list(ep_tt_dict.keys())
            tts = list(ep_tt_dict.values())
            smoothed_tts = running_average(tts, last_n=self.smooth_by)
            plt.plot(episodes, smoothed_tts, color=self.colors[idx], label=kind)

        for phase_idx, phase in enumerate(self.phases):
            color = self.phase_colors[phase_idx % len(self.phase_colors)]
            plt.axvline(x=phase, label=self.phase_names[phase_idx], linestyle='--', color=color)

        plt.xlabel('Episode')
        plt.ylabel('Mean Travel Time')
        plt.grid(True, axis='y')
        plt.title('Mean Travel Times Over Episodes')
        plt.legend()

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Travel times are saved to {save_to}")

####################
    

#################### TRAVEL TIME DISTRIBUTIONS
    
    def visualize_tt_distributions(self):
        save_to = make_dir(self.params[kc.PLOTS_FOLDER], self.params[kc.TT_DIST_PLOT_FILE_NAME])
    
        num_rows, num_cols = 2, 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(self.multimode_width * num_cols, self.multimode_height * num_rows))
        fig.tight_layout(pad=5.0)

        if num_rows > 1:   axes = axes.flatten()   # Flatten axes

        # Plot mean travel times for each OD
        mean_tt_od = self._retrieve_data_per_od(kc.TRAVEL_TIME, transform='mean')
        sorted_keys = sorted(mean_tt_od.keys())
        for idx, od in enumerate(sorted_keys):
            episodes = list(mean_tt_od[od].keys())
            mean_tt = list(mean_tt_od[od].values())
            smoothed_tt = running_average(mean_tt, last_n=self.smooth_by)
            axes[0].plot(episodes, smoothed_tt, color=self.colors[idx], label=od)
        for phase_idx, phase in enumerate(self.phases):
            color = self.phase_colors[phase_idx % len(self.phase_colors)]
            axes[0].axvline(x=phase, label=self.phase_names[phase_idx], linestyle='--', color=color)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Mean Travel Time')
        axes[0].grid(True, axis='y')
        axes[0].set_title('Mean Human Travel Times Per OD Over Episodes')
        axes[0].legend()


        # Plot variance travel times for all, humans and machines
        variance_travel_times = self._retrieve_data_per_kind(kc.TRAVEL_TIME, transform='variance')
        for idx, (kind, ep_tt_dict) in enumerate(variance_travel_times.items()):
            episodes = list(ep_tt_dict.keys())
            var_tts = list(ep_tt_dict.values())
            smoothed_var_tts = running_average(var_tts, last_n=self.smooth_by)
            axes[1].plot(episodes, smoothed_var_tts, color=self.colors[idx], label=kind)

        for phase_idx, phase in enumerate(self.phases):
            color = self.phase_colors[phase_idx % len(self.phase_colors)]
            axes[1].axvline(x=phase, label=self.phase_names[phase_idx], linestyle='--', color=color)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Variance')
        axes[1].grid(True, axis='y')
        axes[1].set_title('Variance Travel Times Over Episodes')
        axes[1].legend()

        # Plot boxplot and violinplot for rewards
        all_travel_times = self._retrieve_data_per_kind(kc.TRAVEL_TIME)
        eps_to_plot = [ep-1 for ep in self.phases[1:]] + [self.saved_episodes[-1]]
        data_to_plot = [all_travel_times[kc.TYPE_HUMAN][ep] for ep in eps_to_plot]
        labels = [f'Humans ({ph})' for ph in self.phase_names]

        bplot = axes[2].boxplot(data_to_plot, labels=labels, patch_artist=True)
        for idx, (patch, med) in enumerate(zip(bplot['boxes'], bplot['medians'])):
            color = self.phase_colors[idx]
            patch.set_facecolor(color)
            med.set_color('black')
            med.set_linewidth(2)
        axes[2].grid(axis = 'y')
        axes[2].set_ylabel('Travel Times')
        axes[2].set_title(f'Human Travel Time Distributions (End of Each Phase)')


        dark_gray = '#333333'
        axes[3].set_facecolor(dark_gray)
        for idx, (label, data) in enumerate(zip(labels, data_to_plot)):
            data = np.array(data)
            data[np.isinf(data)] = np.nan  # Convert inf to NaN

            sns.kdeplot(data, ax=axes[3], label=label, alpha=0.8, fill=True, linewidth=3, color=self.colors[idx], clip=(0, None))
            median_val, mean_val = np.nanmedian(data), np.nanmean(data)
            # Plot a vertical line from top to mid-plot for median
            axes[3].axvline(median_val, color=self.colors[idx], linestyle='-', linewidth=2, ymin=0.5, ymax=1, label=f'Median {label}')
            # Plot a vertical line from bottom to mid-plot for mean
            axes[3].axvline(mean_val, color=self.colors[idx], linestyle='--', linewidth=2, ymin=0, ymax=0.5, label=f'Mean {label}')
        axes[3].set_xlim(0, None)
        axes[3].set_xlabel('Travel Times')
        axes[3].set_ylabel('Probability Density')
        axes[3].set_title(f'Human Travel Time Distributions (End of Each Phase)')
        axes[3].legend()

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Travel time distributions are saved to {save_to}")

####################
    

#################### ACTIONS

    def visualize_actions(self):
        save_to = make_dir(self.params[kc.PLOTS_FOLDER], self.params[kc.ACTIONS_PLOT_FILE_NAME])

        all_actions = self._retrieve_data_per_od(kc.ACTION)
        unique_actions = {od: set([item for sublist in val.values() for item in sublist]) for od, val in all_actions.items()}
        all_actions = {od : [Counter(a) for a in val.values()] for od, val in all_actions.items()}
        num_od_pairs = len(all_actions)
        
        # Determine the layout of the subplots (rows x columns)
        num_columns = self.default_num_columns
        num_rows = (num_od_pairs + num_columns - 1) // num_columns  # Calculate rows needed
        
        figure_size = (self.multimode_width * num_columns, self.multimode_height * num_rows)
        fig, axes = plt.subplots(num_rows, num_columns, figsize=figure_size)
        fig.tight_layout(pad=5.0)
        
        if num_rows > 1:   axes = axes.flatten()   # Flatten axes

        for idx, (od, actions) in enumerate(all_actions.items()):
            ax = axes[idx]
            for idx2, unique_action in enumerate(unique_actions[od]):
                action_data = [ep_actions.get(unique_action, 0) for ep_actions in actions]
                action_data = running_average(action_data, last_n=self.smooth_by)
                ax.plot(self.saved_episodes, action_data, color=self.colors[idx2], label=f"{unique_action}")

            for phase_idx, phase in enumerate(self.phases):
                color = self.phase_colors[phase_idx % len(self.phase_colors)]
                ax.axvline(x=phase, label=self.phase_names[phase_idx], linestyle='--', color=color)

            ax.set_title(f"Actions for {od}")
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Number of Drivers')
            ax.grid(True, axis='y')
            ax.legend()

        # Hide unused subplots if any
        for ax in axes[idx+1:]:   ax.axis('off')    # Hide unused subplots if any

        for ax in axes.flat:    ax.legend().set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=num_columns)
        fig.subplots_adjust(top=0.90)

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Actions are saved to {save_to}")
            
####################
    

#################### ACTION SHIFTS
    
    def visualize_action_shifts(self):
        save_to = make_dir(self.params[kc.PLOTS_FOLDER], self.params[kc.ACTIONS_SHIFTS_PLOT_FILE_NAME])

        all_od_pairs = self._retrieve_all_od_pairs()
        all_od_pairs = sorted(all_od_pairs, key=lambda x: f"{x[0]}-{x[1]}")
        
        # Determine the layout of the subplots (rows x columns)
        num_columns = self.default_num_columns
        num_rows = (len(all_od_pairs) + num_columns - 1) // num_columns  # Calculate rows needed
        
        figure_size = (self.multimode_width * num_columns, self.multimode_height * num_rows)
        fig, axes = plt.subplots(num_rows, num_columns, figsize=figure_size)
        fig.tight_layout(pad=5.0)
        
        if num_rows > 1:   axes = axes.flatten()   # Flatten axes

        for idx, od in enumerate(all_od_pairs):
            ax = axes[idx]
            origin, destination = od

            all_actions, unique_actions = self._retrieve_selected_actions(origin, destination)

            for idx2, (kind, ep_to_actions) in enumerate(all_actions.items()):

                episodes = list(ep_to_actions.keys())

                for idx3, action in enumerate(unique_actions):
                    action_data = [ep_counter[action] / sum(ep_counter.values()) for ep_counter in all_actions[kind].values()]
                    action_data = running_average(action_data, last_n=self.smooth_by)
                    color = self.colors[idx3]
                    linestyle = self.linestyles[idx2 % len(self.linestyles)]
                    ax.plot(episodes, action_data, color=color, linestyle=linestyle, label=f"{kind}-{action}")

            for phase_idx, phase in enumerate(self.phases):
                color = self.phase_colors[phase_idx % len(self.phase_colors)]
                ax.axvline(x=phase, label=self.phase_names[phase_idx], linestyle='--', color=color)

            ax.set_xlabel('Episodes')
            ax.set_ylabel('Number of Drivers (Scaled by Group Size)')
            ax.grid(True, axis='y')
            ax.set_title(f'Actions for {od}')
            ax.legend()

        for ax in axes[idx+1:]:   ax.axis('off')    # Hide unused subplots if any

        for ax in axes.flat:    ax.legend().set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=num_columns)
        fig.subplots_adjust(top=0.85)

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Actions shifts are saved to {save_to}")
    
####################
    

#################### SIM LENGTH
    
    def visualize_sim_length(self):
        save_to = make_dir(self.params[kc.PLOTS_FOLDER], self.params[kc.SIMULATION_LENGTH_PLOT_FILE_NAME])

        sim_lengths = self._retrieve_sim_length()
        sim_lengths = running_average(sim_lengths, last_n=self.smooth_by)

        plt.figure(figsize=(self.default_width, self.default_height))
        plt.plot(self.saved_episodes, sim_lengths, color=self.colors[0], label="Simulation timesteps")
        for phase_idx, phase in enumerate(self.phases):
            color = self.phase_colors[phase_idx % len(self.phase_colors)]
            plt.axvline(x=phase, label=self.phase_names[phase_idx], linestyle='--', color=color)
        plt.xlabel('Episode')
        plt.ylabel('Simulation Length')
        plt.title('Simulation Length Over Episodes')
        plt.legend()

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Simulation lengths are saved to {save_to}")



    def _retrieve_sim_length(self):
        latest_arrivals = list()
        for episode in self.saved_episodes:
            data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            data = pd.read_csv(data_path)
            arrival_times = data.apply(lambda row: row[kc.AGENT_START_TIME] + row[kc.TRAVEL_TIME], axis=1)
            latest_arrival = max(arrival_times)
            latest_arrivals.append(latest_arrival)
        return latest_arrivals
    
####################


#################### LOSSES

    def visualize_losses(self):
        save_to = make_dir(self.params[kc.PLOTS_FOLDER], self.params[kc.LOSSES_PLOT_FILE_NAME])

        losses = self._retrieve_losses()
        if not losses: return
        losses = running_average(losses, last_n=self.smooth_by)

        plt.figure(figsize=(self.default_width, self.default_height))
        plt.plot(losses, color=self.colors[0])
        plt.xlabel('Training Progress (Episodes)')
        plt.ylabel('MSE Loss (Log Scale)')
        plt.title('Mean MSE Loss Over Training Progress for AVs')
        plt.yscale('log')
        plt.grid(True, axis='y')

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Losses are saved to {save_to}")



    def _retrieve_losses(self):
        losses = list()
        if not os.path.isfile(self.loss_file_path): return None
        with open(self.loss_file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                losses.append(float(line.strip()))
        return losses
    
####################


#################### HELPERS

    def _retrieve_data_per_kind(self, data_key, transform=None):
        all_values_dict = {kc.ALL : dict()}
        for episode in self.saved_episodes:
            data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            data = pd.read_csv(data_path)
            kinds, values, found_kinds = data[kc.AGENT_KIND], data[data_key], set(data[kc.AGENT_KIND])
            for kind in found_kinds:
                if kind not in all_values_dict:    all_values_dict[kind] = dict()
            values_per_kind =  {k : list() for k in found_kinds}
            for kind, value in zip(kinds, values):
                values_per_kind[kind].append(value)
            values_per_kind[kc.ALL] = [item for sublist in values_per_kind.values() for item in sublist]
            for kind in values_per_kind:
                if transform == 'mean':
                    all_values_dict[kind][episode] = mean(values_per_kind[kind])
                elif transform == 'variance':
                    all_values_dict[kind][episode] = variance(values_per_kind[kind])
                else:
                    all_values_dict[kind][episode] = values_per_kind[kind]
        return all_values_dict
    

    def _retrieve_data_per_od(self, data_key, transform=None):
        all_od_pairs = self._retrieve_all_od_pairs()
        od_to_key = lambda o, d: f"{o} - {d}"
        all_values_dict = {od_to_key(od[0], od[1]) : dict() for od in all_od_pairs}
        for episode in self.saved_episodes:
            data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            episode_data = pd.read_csv(data_path)
            episode_origins, episode_destinations, values = episode_data[kc.AGENT_ORIGIN], episode_data[kc.AGENT_DESTINATION], episode_data[data_key]
            values_per_od = {od_to_key(od[0], od[1]): list() for od in all_od_pairs}
            for idx, value in enumerate(values):
                values_per_od[od_to_key(episode_origins[idx], episode_destinations[idx])].append(value)
            for od in values_per_od:
                if transform == 'mean':
                    all_values_dict[od][episode] = mean(values_per_od[od])
                elif transform == 'variance':
                    all_values_dict[od][episode] = variance(values_per_od[od])
                else:
                    all_values_dict[od][episode] = values_per_od[od]
        return all_values_dict
        
    
    def _retrieve_selected_actions(self, origin, destination):
        all_actions = dict()
        unique_actions = set()
        for episode in self.saved_episodes:
            data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            data = pd.read_csv(data_path)
            data = data[(data[kc.AGENT_ORIGIN] == origin) & (data[kc.AGENT_DESTINATION] == destination)]
            kinds, actions = data[kc.AGENT_KIND], data[kc.ACTION]
            for kind in set(kinds):
                if kind not in all_actions:    all_actions[kind] = dict()
            unique_actions.update(actions)
            actions_counters = {k : Counter() for k in set(kinds)}
            for kind, action in zip(kinds, actions):
                actions_counters[kind][action] += 1
            for kind in actions_counters.keys():
                all_actions[kind][episode] = actions_counters[kind]
        return all_actions, unique_actions


    def _retrieve_all_od_pairs(self):
        all_od_pairs = list()
        data_path = os.path.join(self.episodes_folder, f"ep{self.saved_episodes[0]}.csv")
        episode_data = pd.read_csv(data_path)
        episode_data = episode_data[[kc.AGENT_ORIGIN, kc.AGENT_DESTINATION]]
        for _, row in episode_data.iterrows():
            origin, destination = row[kc.AGENT_ORIGIN], row[kc.AGENT_DESTINATION]
            all_od_pairs.append((origin, destination))
        all_od_pairs = list(set(all_od_pairs))
        return all_od_pairs
    
####################




def plotter(params):
    plotter = Plotter(params)
    plotter.plot()
    return plotter