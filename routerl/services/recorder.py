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
        params (list): A list of parameters.
    Attributes:
        records_folder: RECORDS_FOLDER
        episodes_folder: EPISODES_LOGS_FOLDER
        detector_folder: DETECTORS_LOGS_FOLDER
        loss_file_path: LOSSES_LOG_FILE_NAME
    Methods:
        _clear_records: removes records from records_folder
        _get_txt_file_path: gets text file from records_folder
        record: remembers episodes and detectors
        remember_episode: remembers episode
        remember_detector: remembers detector
        save_losses: saves losses
    """

    def __init__(self, params):
        self.params = params
        self.records_folder = self.params[kc.RECORDS_FOLDER]

        self.episodes_folder = make_dir([self.records_folder, kc.EPISODES_LOGS_FOLDER])
        self.detector_folder = make_dir([self.records_folder, kc.DETECTOR_LOGS_FOLDER])

        self._clear_records(self.episodes_folder)
        self._clear_records(self.detector_folder)
        
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

    def _get_txt_file_path(self, filename) -> None:
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

    def record(self, episode, ep_observations, rewards, cost_tables, det_dict) -> None:
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

        self.remember_episode(episode, ep_observations, rewards, cost_tables)
        self.remember_detector(episode, det_dict) ## pass self.det_dict
        # TODO more to add

    def remember_episode(self, episode, ep_observations, rewards, cost_tables) -> None:
        """Remember the episode.

        Args:
            episode: episode.
            ep_observations: episode observations.
            rewards: rewards.
            cost_tables: cost_tables.
        Returns:
            None
        """
        
        ep_observations_df = pl.from_dicts(ep_observations)
        rewards_df = pl.from_dicts(rewards)

        for entry in cost_tables:
            entry['cost_table'] = ','.join(map(str, entry['cost_table']))

        cost_tables_df = pl.from_dicts(cost_tables)

        merged_df = ep_observations_df.join(rewards_df, on=kc.AGENT_ID)
        merged_df = merged_df.join(cost_tables_df, on=kc.AGENT_ID)
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