import random
import traci

from keychain import Keychain as kc

    
class SumoController:

    """
    Class to control SUMO
    """

    def __init__(self, params):
        self.sumo_type = params[kc.SUMO_TYPE]
        self.config = params[kc.SUMO_CONFIG_PATH]
        self.label = f"{random.randint(0, 10000)}"
    
    def sumo_start(self):
        sumo_binary = self.sumo_type
        sumo_cmd = [sumo_binary, "-c", self.config]
        traci.start(sumo_cmd, label=self.label)

    def sumo_stop(self):
        traci.switch(self.label)
        traci.close()

    def sumo_reset(self):
        traci.switch(self.label)
        traci.load(['-c', self.config])