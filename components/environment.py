from abc import ABC, abstractmethod

from .simulator import SumoSimulator
from keychain import Keychain as kc

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
        self.episode_actions = dict()
        
        print("[SUCCESS] Environment initiated!")

    #####################

    ##### CONTROL #####

    def start(self):
        self.simulator.start()

    def stop(self):
        self.simulator.stop()

    def reset(self):
        self.episode_actions = dict()
        self.simulator.reset()

    #####################

    ##### EPISODE OPS #####
    
    def get_observation(self):
        return self.simulator.timestep, self.episode_actions.values()


    def step(self, actions):
        step_actions = {agent.id : self._get_action_data(agent, action) for agent, action in actions}
        self.episode_actions.update(step_actions)
        timestep, arrivals = self.simulator.step(step_actions)
        travel_times = {int(veh_id): {kc.TRAVEL_TIME : (timestep - self.episode_actions[int(veh_id)][kc.AGENT_START_TIME]) / 60.0} for veh_id in arrivals}
        for key in travel_times:    travel_times[key].update(self.episode_actions[key])
        return travel_times.values()
    
    
    def _get_action_data(self, agent, action):
        action_data = [agent.id, agent.kind, action, agent.origin, agent.destination, agent.start_time]
        return {k: v for k, v in zip(self.action_cols, action_data)}
    
    #####################

    ##### DATA #####

    def get_free_flow_times(self):
        free_flow_times = self.simulator.get_free_flow_times()
        return free_flow_times

    #####################