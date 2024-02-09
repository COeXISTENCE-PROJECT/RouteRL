import traci

from keychain import Keychain as kc

    
class SumoController:

    """
    Class to control SUMO
    """

    def __init__(self, params):
        self.sumo_type = params[kc.SUMO_TYPE]
        self.config = params[kc.SUMO_CONFIG_PATH]
    
    def sumo_start(self):
        sumo_binary = self.sumo_type
        sumo_cmd = [sumo_binary, "-c", self.config]
        traci.start(sumo_cmd)

    def sumo_stop(self):
        traci.close()

    def sumo_reset(self):
        traci.load(['-c', self.config])