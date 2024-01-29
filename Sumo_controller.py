from keychain import Keychain as kc
import traci

class Sumo:

    def __init__(self,params):
        self.sumo_type = params[kc.SIMULATION_PARAMETERS][kc.SUMO_TYPE]
        self.config = params[kc.SIMULATION_PARAMETERS][kc.SUMO_CONFIG_PATH]
    
    def Sumo_start(self):
        sumo_binary = self.sumo_type
        sumo_cmd = [sumo_binary, "-c", self.config]
        traci.start(sumo_cmd)

    def Sumo_stop(self):
        traci.close()