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
        self.episode_actions = pd.DataFrame(columns = self.action_cols)
        
        self.timestep = 0

        print("[SUCCESS] Environment initiated!")

    #####################

    ##### CONTROL #####

    def start(self):
        self.simulator.start()

    def stop(self):
        self.simulator.stop()

    def reset(self):
        self.episode_actions = pd.DataFrame(columns = self.action_cols)
        self.simulator.reset()
        self.timestep = 0

    #####################

    ##### EPISODE OPS #####
    
    def get_observation(self):
        return self.timestep, self.episode_actions

    def step(self, actions):
        
        action_data_list = list()
        for agent, action in actions:
            action_data = [agent.id, agent.kind, action, agent.origin, agent.destination, agent.start_time]
            action_data_list.append({key : value for key, value in zip(self.action_cols, action_data)})
        step_actions = pd.DataFrame(action_data_list)
        
        self.timestep, travel_times = self.simulator.step(step_actions)
        
        if not travel_times.empty:
            travel_times = travel_times.merge(self.episode_actions, how='left', on=kc.AGENT_ID)
        else:
            cols = list(set(list(travel_times.columns) + list(self.episode_actions.columns)))
            travel_times = pd.DataFrame(columns = cols)
            
        self.episode_actions = pd.concat([self.episode_actions, step_actions])
        return travel_times
    
    #####################

    ##### DATA #####

    def get_free_flow_times(self):
        free_flow_times = self.simulator.get_free_flow_times()
        return free_flow_times

    #####################