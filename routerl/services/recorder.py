import logging
import os
import polars as pl

from routerl.keychain import Keychain as kc
from routerl.utilities import make_dir

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class Recorder:
    """Record the training process.

    Args:
        params (list):
            Plotter parameters as specified `here <https://coexistence-project.github.io/RouteRL/documentation/pz_env.html#>`_.
        
    Methods:
        record: saves episode data and detector statistics to disk.
        remember_episode: records episode data.
        remember_detector: decords detector data
    """

    def __init__(self, params):
        self.params = params
        self.records_folder = self.params[kc.RECORDS_FOLDER]

        self.episodes_folder = make_dir([self.records_folder, kc.EPISODES_LOGS_FOLDER])
        self.detector_folder = make_dir([self.records_folder, kc.DETECTOR_LOGS_FOLDER])
        self.sumo_folder = make_dir([self.records_folder, kc.SUMO_LOGS_FOLDER])
        self.marginal_cost_folder = make_dir([self.records_folder, kc.MARGINAL_COST_MATRIX])

        self._clear_records(self.episodes_folder)
        self._clear_records(self.detector_folder)
        self._clear_records(self.sumo_folder)
        self._clear_records(self.marginal_cost_folder)
        
        self.loss_file_path = self._get_txt_file_path(kc.LOSSES_LOG_FILE_NAME)
        logging.info(f"[SUCCESS] Recorder is now here to record!")

    ################################
    ######## Initial helpers #######
    ################################

    def _clear_records(self, folder) -> None:
        """Clears records from records_folder.

        Args:
            folder: folder to clear records from.
        Returns:
            None
        """

        if os.path.exists(folder):
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))

    def _get_txt_file_path(self, filename) -> str:
        """Gets text file from records_folder.

        Args:
            filename: filename.
        Returns:
            None
        """

        log_file_path = make_dir(self.records_folder, filename)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        return log_file_path

    ################################
    ####### Remember methods #######
    ################################

    def record(self, episode, ep_observations, cost_tables, det_dict) -> None:
        """Records the episode.

        Args:
            episode: episode.
            ep_observations: episode observations.
            rewards: rewards.
            cost_tables: cost_tables.
            det_dict: det_dict.
        Returns:
            None
        """

        self.remember_episode(episode, ep_observations, cost_tables)
        self.remember_detector(episode, det_dict)
        

    def remember_episode(self, episode, ep_observations, cost_tables) -> None:
        """Remember the episode.

        Args:
            episode: episode.
            ep_observations: episode observations.
            cost_tables: cost_tables.
        Returns:
            None
        """
        
        ep_observations_df = pl.from_dicts(ep_observations)

        for entry in cost_tables:
            entry['cost_table'] = ','.join(map(str, entry['cost_table']))

        cost_tables_df = pl.from_dicts(cost_tables)
        
        merged_df = ep_observations_df.join(cost_tables_df, on=kc.AGENT_ID)
        merged_df.write_csv(make_dir(self.episodes_folder, f"ep{episode}.csv"))

    def remember_detector(self, episode, det_dict) -> None:
        """Remember the detector.

        Args:
            episode: episode.
            det_dict: det_dict.
        Returns:
            None
        """
        
        df = pl.DataFrame(list(det_dict.items()), schema=['detid', 'flow'], orient="row")
        df.write_csv(make_dir(self.detector_folder, f'detector_ep{episode}.csv'))

    def save_losses(self, agents) -> None:
        """Save losses.

        Args:
            agents: agents.
        Returns:
            None
        """

        losses = list()
        for a in agents:
            loss = getattr(a.model, 'loss', None)
            if loss is not None:
                losses.append(loss)
        if len(losses):
            mean_losses = [0] * len(losses[-1])
            for loss in losses:
                for i, l in enumerate(loss):
                    mean_losses[i] += l
            mean_losses = [m / len(losses) for m in mean_losses]
            with open(self.loss_file_path, "w") as file:
                for m_l in mean_losses:
                    file.write(f"{m_l}\n")


    def remember_marginal_costs(self, marginal_cost_calculation: dict, episode: int, machine_agents: list) -> None:
        """Savr the marginal cost matrices

        Args:
            marginal_cost_calculation: dictionary that contains the cost of each agent to each agent
            episode: episode, 
            machine_agents: machine_agents
        """
        # Save the agents based on their start time
        sorted_agents = sorted(machine_agents, key=lambda agent: agent.start_time)

        sorted_ids = [agent.id for agent in sorted_agents]
        sorted_machine_names = [f"Machine {id_}" for id_ in sorted_ids]

        formatted_rows = []
        for row_id in sorted_ids:
            row_label = f"Machine {row_id}"
            row_data = marginal_cost_calculation.get(row_id, {})
            cleaned_row_data = {str(k): v for k, v in row_data.items()}

            # Fill in all columns in the sorted order, use None if missing
            full_row = {col: cleaned_row_data.get(col, None) for col in sorted_machine_names}
            full_row["ID"] = row_label
            formatted_rows.append(full_row)

        pl_df = pl.DataFrame(formatted_rows)

        column_order = ["ID"] + [col for col in sorted_machine_names if col in pl_df.columns]
        if "ID" in pl_df.columns:
            pl_df = pl_df.select(column_order)

        filename = f"marginal_cost_matrix_{episode}.csv"
        pl_df.write_csv(make_dir(self.marginal_cost_folder, filename))

        return