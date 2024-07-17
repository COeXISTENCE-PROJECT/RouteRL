import logging
import os
import polars as pl
import pandas as pd

from keychain import Keychain as kc
from utilities import make_dir

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

class Recorder:

    """
    Record the training process.
    """

    def __init__(self):

        self.episodes_folder = make_dir([kc.RECORDS_FOLDER, kc.EPISODES_LOGS_FOLDER])
        self.detector_folder = make_dir([kc.RECORDS_FOLDER, kc.DETECTOR_LOGS_FOLDER])

        self._clear_records(self.episodes_folder)
        self._clear_records(self.detector_folder)
        
        self.sim_length_file_path = self._get_txt_file_path(kc.SIMULATION_LENGTH_LOG_FILE_NAME)
        self.loss_file_path = self._get_txt_file_path(kc.LOSSES_LOG_FILE_NAME)
        logging.info(f"[SUCCESS] Recorder is now here to record!")

#################### INIT HELPERS

    def _clear_records(self, folder):
        if os.path.exists(folder):
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))
    

    def _get_txt_file_path(self, filename):
        log_file_path = make_dir(kc.RECORDS_FOLDER, filename)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        return log_file_path
    
####################
        

#################### REMEMBER FUNCTIONS
    
    def record(self, episode, ep_observations, rewards, det_dict):
        self.remember_episode(episode, ep_observations, rewards)
        self.remember_detector(episode, det_dict) ## pass self.det_dict
        # more to add


    def remember_episode(self, episode, ep_observations, rewards):
        ep_observations_df = pl.from_dicts(ep_observations)
        rewards_df = pl.from_dicts(rewards)
        merged_df = ep_observations_df.join(rewards_df, on=kc.AGENT_ID)
        merged_df.write_csv(make_dir(self.episodes_folder, f"ep{episode}.csv"))


    def remember_detector(self,episode, det_dict):
        df = pd.DataFrame(list(det_dict.items()), columns=['detid', 'flow'])
        df.to_csv(make_dir(self.detector_folder, f'detector_ep{episode}.csv'),index=False)


    def save_losses(self, agents):
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

####################
            
    
        