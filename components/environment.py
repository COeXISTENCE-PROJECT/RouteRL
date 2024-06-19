import pandas as pd

from abc import ABC, abstractmethod
from prettytable import PrettyTable

from .simulator import SumoSimulator
from keychain import Keychain as kc
from utilities import make_dir



class BaseEnvironment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self):
        pass



class TrafficEnvironment(BaseEnvironment):
    def __init__(self, params, simulator: SumoSimulator):
        super().__init__()

        self.action_space_size = params[kc.ACTION_SPACE_SIZE]
        self.simulator = simulator

        self.action_cols = [kc.AGENT_ID, kc.AGENT_KIND, kc.ACTION, kc.AGENT_ORIGIN, kc.AGENT_DESTINATION, kc.AGENT_START_TIME]
        self.actions = pd.DataFrame(columns = self.action_cols)
        self.all_actions = pd.DataFrame(columns = self.action_cols)
        
        self.timestep = 0

        print("[SUCCESS] Environment initiated!")

    #####################

    ##### CONTROL #####

    def start(self):
        self.simulator.start()

    def stop(self):
        self.simulator.stop()

    def reset(self):
        self.actions = pd.DataFrame(columns = self.action_cols)
        self.all_actions = pd.DataFrame(columns = self.action_cols)
        self.simulator.reset()
        self.timestep = 0

    #####################

    ##### EPISODE OPS #####

    def register_action(self, agent, action):
        action_data = [agent.id, agent.kind, action, agent.origin, agent.destination, agent.start_time]
        self.actions.loc[len(self.actions.index)] = {key : value for key, value in zip(self.action_cols, action_data)}
    
    def get_observation(self):
        return self.timestep, self.all_actions

    def step(self):
        self.timestep = self.simulator.step(self.actions)
        self.all_actions = pd.concat([self.all_actions, self.actions])
        self.actions = pd.DataFrame(columns = self.action_cols)
    
    def get_travel_times(self):
        while self.simulator.check_simulation_continues():
            self.step()
        travel_times = self.simulator.get_travel_times()
        travel_times = travel_times.merge(self.all_actions, on=kc.AGENT_ID)
        return travel_times
    
    #####################

    ##### DATA #####

    def get_free_flow_times(self):
        free_flow_times = self.simulator.get_free_flow_times()
        return free_flow_times

    #####################