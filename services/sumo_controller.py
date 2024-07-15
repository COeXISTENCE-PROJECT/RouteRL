import traci

from keychain import Keychain as kc

    
class SumoController: #RK: This definitely needs documentation. We have: environment, Simulator and SumoController - describe differences.

    """
    Class to control SUMO
    """

    def __init__(self, params):
        self.seed=params[kc.SEED]
        self.sumo_type = params[kc.SUMO_TYPE]
        self.config = params[kc.SUMO_CONFIG_PATH]
    
    def sumo_start(self): # RK: why sumo_start - maybe just .start()?
        sumo_binary = self.sumo_type
        sumo_cmd = [sumo_binary,"--seed", self.seed, "-c", self.config] # RK: This string will go longer in future, with possible other sys args, so this shall be populated from config.
        traci.start(sumo_cmd)

    def sumo_stop(self): # RK: why sumo_stop - maybe just .stop()?
        traci.close()

    def sumo_reset(self): # RK: why sumo_reset - maybe just .reset()?
        traci.load(["--seed", self.seed,'-c', self.config])