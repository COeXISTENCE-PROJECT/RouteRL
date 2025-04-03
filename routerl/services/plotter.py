import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings

from collections import Counter
from matplotlib import rcParams
from statistics import mean, variance
from typing import Union

from routerl.keychain import Keychain as kc
from routerl.utilities import make_dir
from routerl.utilities import running_average
from routerl.utilities import get_params

warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class Plotter:
    """Plot the results of the training

    Args:
        params (list): 
            Plotter parameters as specified `here <https://coexistence-project.github.io/RouteRL/documentation/pz_env.html#>`_.
            
    Methods:
        plot: plot the results
        visualise_mean_reward: visualise the mean reward
        visualise_mean_travel_times: visualise the mean travel times
        travel_tt_distributions: visualise the travel time distributions
        visualise_actions: visualise the actions
        visualise_action_shifts: visualise the action shifts
        visualise_sim_length: visualise the simulation length
        visualise_losses: visualise the losses
        
    Note:
        Plotter styling configuration is stored in `routerl/services/plotter_config.json` file.
    """

    def __init__(self, params):
        self.label_size = 12
        self.tick_label_size = 12
        self.line_width = 3
        self.title_size = 16
        self.default_width = 15
        self.default_height = 15
        self.legend_font_size = 12
        self.default_num_columns = 3
        self.multimode_width = 3
        self.multimode_height = 3
        self.linestyles = ['-']
        self.colors = []

        self.params = params
        self.plot_choices = params[kc.PLOT_CHOICES]
        self.phases = params[kc.PHASES]
        self.phase_names = params[kc.PHASE_NAMES]
        self.smooth_by = params[kc.SMOOTH_BY]
        self.records_folder = params[kc.RECORDS_FOLDER]

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), kc.PLOTTER_CONFIG_FILE), 'r') as file:
            config = json.load(file)
        for key, value in config.items():
            setattr(self, key, value)
        self.phase_colors = list(reversed(self.colors))
        rcParams['font.family'] = self.font_family
        
        make_dir(self.records_folder)
        self.episodes_folder = make_dir([self.records_folder, kc.EPISODES_LOGS_FOLDER])
        self.loss_file_path = make_dir(self.records_folder, kc.LOSSES_LOG_FILE_NAME)

        self.saved_episodes = list()

    ################################
    ######## VISUALIZE ALL #########
    ################################

    def plot(self) -> None:
        """Plot the results of the training

        Returns:
            None
        """

        self.saved_episodes = self._get_episodes()
        
        if self.plot_choices == kc.PLOT_ALL:
            self.visualize_mean_rewards()
            self.visualize_mean_travel_times()
            self.visualize_tt_distributions()
            self.visualize_actions()
            self.visualize_action_shifts()
            self.visualize_sim_length()
            self.visualize_losses()
        elif self.plot_choices == kc.PLOT_BASIC:
            self.visualize_mean_rewards()
            self.visualize_mean_travel_times()
        elif self.plot_choices != kc.PLOT_NONE:
            logging.warning(f"Plot choice mode {self.plot_choices} is not recognised. Options: {kc.PLOT_ALL}, {kc.PLOT_BASIC}, {kc.PLOT_NONE}")

    def _get_episodes(self) -> list[int]:
        """Get the episodes data

        Returns:
            sorted_episodes (list[int]): the sorted episodes data
        Raises:
            FileNotFoundError: If the episodes folder does not exist
        """

        eps = list()
        if os.path.exists(self.episodes_folder):
            for file in os.listdir(self.episodes_folder):
                episode = int(file.split('ep')[1].split('.csv')[0])
                eps.append(episode)
        else:
            raise FileNotFoundError(f"Episodes folder does not exist!")
        return sorted(eps)

    ################################
    ########### REWARDS ############
    ################################

    def visualize_mean_rewards(self) -> None:
        """Visualise the mean rewards

        Returns:
            None
        """

        save_to = make_dir(self.params[kc.PLOTS_FOLDER], kc.REWARDS_PLOT_FILE_NAME)
        all_mean_rewards = self._retrieve_data_per_kind(kc.REWARD, transform='mean')

        plt.figure(figsize=(self.default_width, self.default_height), layout='tight')

        for idx, (kind, ep_reward_dict) in enumerate(all_mean_rewards.items()):
            episodes = list(ep_reward_dict.keys())
            rewards = list(ep_reward_dict.values())
            smoothed_rewards = running_average(rewards, last_n=self.smooth_by)
            plt.plot(episodes, smoothed_rewards, color=self.colors[idx], label=kind, linewidth=self.line_width)

        for phase_idx, phase in enumerate(self.phases):
            color = self.phase_colors[phase_idx % len(self.phase_colors)]
            plt.axvline(x=phase,
                        label=self.phase_names[phase_idx],
                        linestyle='--',
                        color=color,
                        linewidth=self.line_width)

        plt.xticks(fontsize=self.tick_label_size)
        plt.yticks(fontsize=self.tick_label_size)
        plt.xlabel('Episode', fontsize=self.label_size)
        plt.ylabel('Mean Reward', fontsize=self.label_size)
        plt.grid(True, axis='y')
        plt.title('Mean Rewards Over Episodes', fontsize=self.title_size, fontweight='bold')
        plt.legend(fontsize=self.legend_font_size)

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Rewards are saved to {save_to}")

    ################################
    ######### TRAVEL TIMES #########
    ################################

    def visualize_mean_travel_times(self) -> None:
        """Visualise the mean travel times

        Returns:
            None
        """

        save_to = make_dir(self.params[kc.PLOTS_FOLDER], kc.TRAVEL_TIMES_PLOT_FILE_NAME)
        all_mean_tt = self._retrieve_data_per_kind(kc.TRAVEL_TIME, transform='mean')

        plt.figure(figsize=(self.default_width, self.default_height), layout='tight')

        for idx, (kind, ep_tt_dict) in enumerate(all_mean_tt.items()):
            episodes = list(ep_tt_dict.keys())
            tts = list(ep_tt_dict.values())
            smoothed_tts = running_average(tts, last_n=self.smooth_by)
            plt.plot(episodes, smoothed_tts, color=self.colors[idx], label=kind, linewidth=self.line_width)

        for phase_idx, phase in enumerate(self.phases):
            color = self.phase_colors[phase_idx % len(self.phase_colors)]
            plt.axvline(x=phase,
                        label=self.phase_names[phase_idx],
                        linestyle='--',
                        color=color,
                        linewidth=self.line_width)

        plt.xticks(fontsize=self.tick_label_size)
        plt.yticks(fontsize=self.tick_label_size)
        plt.xlabel('Episode', fontsize=self.label_size)
        plt.ylabel('Mean Travel Time', fontsize=self.label_size)
        plt.grid(True, axis='y')
        plt.title('Mean Travel Times Over Episodes', fontsize=self.title_size, fontweight='bold')
        plt.legend(fontsize=self.legend_font_size)

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Travel times are saved to {save_to}")

    ################################
    ### TRAVEL TIME DISTRIBUTIONS ##
    ################################

    def visualize_tt_distributions(self) -> None:
        """Visualise the travel time distributions

        Returns:
            None
        """
        save_to = make_dir(self.params[kc.PLOTS_FOLDER], kc.TT_DIST_PLOT_FILE_NAME)
    
        num_rows, num_cols = 2, 2
        fig, axes = plt.subplots(num_rows,
                                 num_cols,
                                 figsize=(self.multimode_width * num_cols,
                                          self.multimode_height * num_rows))
        fig.tight_layout(pad=5.0)

        if num_rows > 1:
            axes = axes.flatten()   # Flatten axes
        if not hasattr(axes, '__getitem__'):
            axes = np.array([axes])  # If only one subplot

        # Plot mean travel times for each OD
        mean_tt_od = self._retrieve_data_per_od(kc.TRAVEL_TIME, transform='mean')
        sorted_keys = sorted(mean_tt_od.keys())
        for idx, od in enumerate(sorted_keys):
            episodes = list(mean_tt_od[od].keys())
            mean_tt = list(mean_tt_od[od].values())
            smoothed_tt = running_average(mean_tt, last_n=self.smooth_by)
            axes[0].plot(episodes, smoothed_tt, color=self.colors[idx], label=od, linewidth=self.line_width)
        for phase_idx, phase in enumerate(self.phases):
            color = self.phase_colors[phase_idx % len(self.phase_colors)]
            axes[0].axvline(x=phase,
                            label=self.phase_names[phase_idx],
                            linestyle='--',
                            color=color,
                            linewidth=self.line_width)
        axes[0].tick_params(axis='both', which='major', labelsize=self.tick_label_size)
        axes[0].set_xlabel('Episode', fontsize=self.label_size)
        axes[0].set_ylabel('Mean Travel Time', fontsize=self.label_size)
        axes[0].grid(True, axis='y')
        axes[0].set_title('Mean Human Travel Times Per OD', fontsize=self.title_size, fontweight='bold')
        axes[0].legend()

        # Plot variance travel times for all, humans and machines
        variance_travel_times = self._retrieve_data_per_kind(kc.TRAVEL_TIME, transform='variance')
        for idx, (kind, ep_tt_dict) in enumerate(variance_travel_times.items()):
            episodes = list(ep_tt_dict.keys())
            var_tts = list(ep_tt_dict.values())
            smoothed_var_tts = running_average(var_tts, last_n=self.smooth_by)
            axes[1].plot(episodes,
                         smoothed_var_tts,
                         color=self.colors[idx],
                         label=kind,
                         linewidth=self.line_width)

        for phase_idx, phase in enumerate(self.phases):
            color = self.phase_colors[phase_idx % len(self.phase_colors)]
            axes[1].axvline(x=phase,
                            label=self.phase_names[phase_idx],
                            linestyle='--',
                            color=color,
                            linewidth=self.line_width)
        axes[1].tick_params(axis='both', which='major', labelsize=self.tick_label_size)
        axes[1].set_xlabel('Episode', fontsize=self.label_size)
        axes[1].set_ylabel('Variance', fontsize=self.label_size)
        axes[1].grid(True, axis='y')
        axes[1].set_title('Variance Travel Times', fontsize=self.title_size, fontweight='bold')
        axes[1].legend()

        if len(self.phases) > 1:
            # Plot boxplot and violin plot for rewards
            all_travel_times = self._retrieve_data_per_kind(kc.TRAVEL_TIME)
            eps_to_plot = list()
            for phase in self.phases[1:]:
                eps_before = [ep for ep in self.saved_episodes if ep < phase]
                eps_to_plot.append(max(eps_before))
            eps_to_plot.append(self.saved_episodes[-1])
            data_to_plot = [all_travel_times[kc.TYPE_HUMAN][ep] for ep in eps_to_plot]
            
            if data_to_plot:
                labels = [f'Humans\n({ph})' for ph in self.phase_names]
                bplot = axes[2].boxplot(data_to_plot, labels=labels, patch_artist=True)
                for idx, (patch, med) in enumerate(zip(bplot['boxes'], bplot['medians'])):
                    color = self.colors[idx]
                    patch.set_facecolor(color)
                    med.set_color('black')
                    med.set_linewidth(2)
                axes[2].tick_params(axis='both', which='major', labelsize=self.tick_label_size)
                axes[2].grid(axis = 'y')
                axes[2].set_ylabel('Travel Times', fontsize=self.label_size)
                axes[2].set_title(f'Human T.T. Distributions (End of Phases)', fontsize=self.title_size, fontweight='bold')

            dark_gray = '#333333'
            axes[3].set_facecolor(dark_gray)
            for idx, (label, data) in enumerate(zip(labels, data_to_plot)):
                data = np.array(data)
                data[np.isinf(data)] = np.nan  # Convert inf to NaN

                sns.kdeplot(data,
                            ax=axes[3],
                            label=label,
                            alpha=0.8,
                            fill=True,
                            linewidth=3,
                            color=self.colors[idx],
                            clip=(0, None))
                median_val, mean_val = np.nanmedian(data), np.nanmean(data)
                # Plot a vertical line from top to mid-plot for median
                axes[3].axvline(median_val,
                                color=self.colors[idx],
                                linestyle='-',
                                linewidth=2,
                                ymin=0.5,
                                ymax=1,
                                label=f'Median {label}')
                # Plot a vertical line from bottom to mid-plot for mean
                axes[3].axvline(mean_val,
                                color=self.colors[idx],
                                linestyle='--',
                                linewidth=2,
                                ymin=0,
                                ymax=0.5,
                                label=f'Mean {label}')
            axes[3].tick_params(axis='both', which='major', labelsize=self.tick_label_size)
            axes[3].set_xlim(0, None)
            axes[3].set_xlabel('Travel Times', fontsize=self.label_size)
            axes[3].set_ylabel('Probability Density', fontsize=self.label_size)
            axes[3].set_title(f'Human T.T. Distributions (End of Phases)', fontsize=self.title_size, fontweight='bold')
            axes[3].legend()

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Travel time distributions are saved to {save_to}")

    ################################
    ########### ACTIONS ############
    ################################

    def visualize_actions(self) -> None:
        """Visualize the actions taken by the agent.

        Returns:
            None
        """

        save_to = make_dir(self.params[kc.PLOTS_FOLDER], kc.ACTIONS_PLOT_FILE_NAME)

        all_actions = self._retrieve_data_per_od(kc.ACTION)
        unique_actions = {
            od: {
                item for sublist in val.values() for item in sublist
            }
            for od, val in all_actions.items()
        }
        all_actions = {od : [Counter(a) for a in val.values()] for od, val in all_actions.items()}
        all_actions = {k: v for k,v in sorted(all_actions.items(), key=lambda x: f"{x[0]}")}
        num_od_pairs = len(all_actions)
        
        # Determine the layout of the subplots (rows x columns)
        num_columns = self.default_num_columns if num_od_pairs > self.default_num_columns else num_od_pairs
        num_rows = (num_od_pairs + num_columns - 1) // num_columns  # Calculate rows needed
        
        figure_size = (self.multimode_width * num_columns, self.multimode_height * num_rows)
        fig, axes = plt.subplots(num_rows, num_columns, figsize=figure_size)
        fig.tight_layout(pad=5.0)
        
        if num_rows > 1:
            axes = axes.flatten()   # Flatten axes
        if not hasattr(axes, '__getitem__'):
            axes = np.array([axes])  # If only one subplot

        for idx, (od, actions) in enumerate(all_actions.items()):
            ax = axes[idx]
            for idx2, unique_action in enumerate(unique_actions[od]):
                action_data = [ep_actions.get(unique_action, 0) for ep_actions in actions]
                action_data = running_average(action_data, last_n=self.smooth_by)
                ax.plot(self.saved_episodes,
                        action_data,
                        color=self.colors[idx2],
                        label=f"{unique_action}",
                        linewidth=self.line_width)

            for phase_idx, phase in enumerate(self.phases):
                color = self.phase_colors[phase_idx % len(self.phase_colors)]
                ax.axvline(x=phase,
                           label=self.phase_names[phase_idx],
                           linestyle='--',
                           color=color,
                           linewidth=self.line_width)

            ax.set_title(f"Actions for {od}", fontsize=self.title_size, fontweight='bold')
            ax.set_xlabel('Episodes', fontsize=self.label_size)
            ax.set_ylabel('Number of Drivers', fontsize=self.label_size)
            ax.grid(True, axis='y')
            ax.legend()
            ax.tick_params(axis='both', which='major', labelsize=self.tick_label_size)

        # Hide unused subplots if any
        for ax in axes[idx+1:]:
            ax.axis('off')

        for ax in axes.flat:
            ax.legend().set_visible(False)
            
        handles, labels = axes[0].get_legend_handles_labels()
        legend_loc = 'upper center' if num_od_pairs > 1 else 'upper right'
        fig.legend(handles, labels, loc=legend_loc, ncol=2, fontsize=self.legend_font_size)
        fig.subplots_adjust(top=0.90)

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Actions are saved to {save_to}")

    ################################
    ######## ACTION SHIFTS #########
    ################################

    def visualize_action_shifts(self) -> None:
        """Visualize the action shifts taken by the agent.

        Returns:
            None
        """

        save_to = make_dir(self.params[kc.PLOTS_FOLDER], kc.ACTIONS_SHIFTS_PLOT_FILE_NAME)

        all_od_pairs = self._retrieve_all_od_pairs()
        all_od_pairs = sorted(all_od_pairs, key=lambda x: f"{x[0]}-{x[1]}")
        
        # Determine the layout of the subplots (rows x columns)
        num_columns = self.default_num_columns if len(all_od_pairs) > self.default_num_columns else len(all_od_pairs)
        num_rows = (len(all_od_pairs) + num_columns - 1) // num_columns  # Calculate rows needed
        
        figure_size = (self.multimode_width * num_columns, self.multimode_height * num_rows)
        fig, axes = plt.subplots(num_rows, num_columns, figsize=figure_size)
        fig.tight_layout(pad=5.0)
        
        if num_rows > 1:   axes = axes.flatten()   # Flatten axes
        if not hasattr(axes, '__getitem__'):    axes = np.array([axes])  # If only one subplot

        for idx, od in enumerate(all_od_pairs):
            ax = axes[idx]
            origin, destination = od

            all_actions, unique_actions = self._retrieve_selected_actions(origin, destination)

            for idx2, (kind, ep_to_actions) in enumerate(all_actions.items()):

                episodes = list(ep_to_actions.keys())

                for idx3, action in enumerate(unique_actions):
                    action_data = [ep_ctr[action] / sum(ep_ctr.values()) for ep_ctr in all_actions[kind].values()]
                    action_data = running_average(action_data, last_n=self.smooth_by)
                    color = self.colors[idx3]
                    linestyle = self.linestyles[idx2 % len(self.linestyles)]
                    ax.plot(episodes,
                            action_data,
                            color=color,
                            linestyle=linestyle,
                            label=f"{kind}-{action}",
                            linewidth=self.line_width)

            for phase_idx, phase in enumerate(self.phases):
                color = self.phase_colors[phase_idx % len(self.phase_colors)]
                ax.axvline(x=phase,
                           label=self.phase_names[phase_idx],
                           linestyle='--',
                           color=color,
                           linewidth=self.line_width)

            ax.set_xlabel('Episodes', fontsize=self.label_size)
            ax.set_ylabel('Fraction of drivers', fontsize=self.label_size)
            ax.grid(True, axis='y')
            ax.set_title(f'Actions for {od}', fontsize=self.title_size, fontweight='bold')
            ax.legend()
            ax.tick_params(axis='both', which='major', labelsize=self.tick_label_size)

        for ax in axes[idx+1:]:
            ax.axis('off')    # Hide unused subplots if any
        for ax in axes.flat:
            ax.legend().set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        legend_loc = 'upper center' if len(all_od_pairs) > 1 else 'upper right'
        fig.legend(handles, labels, loc=legend_loc, ncol=3, fontsize=self.legend_font_size)
        fig.subplots_adjust(top=0.85)

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Actions shifts are saved to {save_to}")

    ################################
    ########## SIM LENGTH ##########
    ################################

    def visualize_sim_length(self) -> None:
        """Visualize the simulation length.

        Returns:
            None
        """

        save_to = make_dir(self.params[kc.PLOTS_FOLDER], kc.SIM_LENGTH_PLOT_FILE_NAME)
        sim_lengths = self._retrieve_sim_length()
        sim_lengths = running_average(sim_lengths, last_n=self.smooth_by)

        plt.figure(figsize=(self.default_width, self.default_height), layout='tight')
        plt.plot(self.saved_episodes,
                 sim_lengths,
                 color=self.colors[0],
                 label="Simulation time steps",
                 linewidth=self.line_width)
        for phase_idx, phase in enumerate(self.phases):
            color = self.phase_colors[phase_idx % len(self.phase_colors)]
            plt.axvline(x=phase,
                        label=self.phase_names[phase_idx],
                        linestyle='--',
                        color=color,
                        linewidth=self.line_width)
        plt.xticks(fontsize=self.tick_label_size)
        plt.yticks(fontsize=self.tick_label_size)
        plt.xlabel('Episode', fontsize=self.label_size)
        plt.ylabel('Simulation Length', fontsize=self.label_size)
        plt.title('Simulation Length Over Episodes', fontsize=self.title_size, fontweight='bold')
        plt.legend(fontsize=self.legend_font_size)

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Simulation lengths are saved to {save_to}")

    def _retrieve_sim_length(self) -> list:
        """Retrieve the simulation length.

        Returns:
            latest arrivals (list): List of the latest arrivals in the simulation.
        """

        latest_arrivals = list()
        for episode in self.saved_episodes:
            data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            data = pd.read_csv(data_path)
            arrival_times = data.apply(lambda row: row[kc.AGENT_START_TIME] + row[kc.TRAVEL_TIME], axis=1)
            latest_arrival = max(arrival_times)
            latest_arrivals.append(latest_arrival)
        return latest_arrivals

    ################################
    ############ LOSES #############
    ################################

    def visualize_losses(self) -> None:
        """Visualize the losses.

        Returns:
            None
        """

        save_to = make_dir(self.params[kc.PLOTS_FOLDER], kc.LOSSES_PLOT_FILE_NAME)

        losses = self._retrieve_losses()
        if not losses: return
        losses = running_average(losses, last_n=self.smooth_by)

        plt.figure(figsize=(self.default_width, self.default_height), layout='tight')
        plt.plot(losses, color=self.colors[0], linewidth=self.line_width)
        plt.xlabel('Training Progress (Episodes)', fontsize=self.label_size)
        plt.ylabel('MSE Loss (Log Scale)', fontsize=self.label_size)
        plt.title('Mean MSE Loss Over Training Progress for AVs', fontsize=self.title_size, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, axis='y')
        plt.xticks(fontsize=self.tick_label_size)
        plt.yticks(fontsize=self.tick_label_size)

        plt.savefig(save_to)
        plt.close()
        logging.info(f"[SUCCESS] Losses are saved to {save_to}")

    def _retrieve_losses(self) -> Union[list[float], None]:
        """Retrieve the losses.

        Returns:
            losses list[float]: List of the losses.
        """

        losses = list()
        if not os.path.isfile(self.loss_file_path):
            return None
        with open(self.loss_file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                losses.append(float(line.strip()))
        return losses

    ################################
    ########### HELPERS ############
    ################################

    def _retrieve_data_per_kind(self, data_key, transform=None) -> dict[str, dict]:
        """Retrieve data per kind.

        Args:
            data_key (str): The key of the data you want to retrieve.
            transform (callable, optional): Optional transform to be applied
        Returns:
            all_values_dict (dict[Any, dict]): Dictionary of all data values per kind.
        """

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
    
    def _retrieve_data_per_od(self, data_key, transform=None) -> dict[str, dict]:
        """Retrieve data per od.

        Args:
            data_key (str): The key of the data you want to retrieve.
            transform (callable, optional): Optional transform to be applied
        Returns:
            all_values_dict (dict[str, dict]): Dictionary of all data values per kind.
        """

        all_od_pairs = self._retrieve_all_od_pairs()
        od_to_key = lambda o, d: f"({o}, {d})"
        all_values_dict = {od_to_key(od[0], od[1]) : dict() for od in all_od_pairs}
        for episode in self.saved_episodes:
            data_path = os.path.join(self.episodes_folder, f"ep{episode}.csv")
            episode_data = pd.read_csv(data_path)
            episode_origins, episode_destinations = episode_data[kc.AGENT_ORIGIN], episode_data[kc.AGENT_DESTINATION]
            values = episode_data[data_key]
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

    def _retrieve_selected_actions(self, origin, destination) -> [dict, dict]:
        """Retrieve selected actions.

        Args:
            origin (str): The origin of the action.
            destination (str): The destination of the action.

        Returns:
            all_actions (dict): Dictionary of all selected actions.
            unique_actions (dict): Dictionary of all unique actions.
        """

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
        """Retrieve all OD pairs.

        Returns:
            all_od_pairs (dict): Dictionary of all OD pairs.
        """

        all_od_pairs = list()
        data_path = os.path.join(self.episodes_folder, f"ep{self.saved_episodes[0]}.csv")
        episode_data = pd.read_csv(data_path)
        episode_data = episode_data[[kc.AGENT_ORIGIN, kc.AGENT_DESTINATION]]
        for _, row in episode_data.iterrows():
            origin, destination = int(row[kc.AGENT_ORIGIN]), int(row[kc.AGENT_DESTINATION])
            all_od_pairs.append((origin, destination))
        all_od_pairs = list(set(all_od_pairs))
        return all_od_pairs


def plotter(params = None):
    """
    Creates a ``Plotter`` instance and plots the results of the training.

    Returns:
        plotter: plotter
    """
    if params is None:
        logging.warning(f"No parameters provided for plotter, "
                        f"using default parameters. This may result in incorrect plots.")
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        params_path = os.path.join(curr_dir, f'../environment/{kc.DEFAULTS_FILE}')
        params = get_params(params_path)
        params = params[kc.PLOTTER]

    plotter = Plotter(params)
    plotter.plot()
    return plotter