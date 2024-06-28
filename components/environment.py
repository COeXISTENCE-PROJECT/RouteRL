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


    def step(self, actions: list[tuple]):
        for agent, action in actions:
            action_dict = {kc.AGENT_ID: agent.id, kc.AGENT_KIND: agent.kind, kc.ACTION: action, \
                kc.AGENT_ORIGIN: agent.origin, kc.AGENT_DESTINATION: agent.destination, kc.AGENT_START_TIME: agent.start_time}
            self.simulator.add_vehice(action_dict)
            self.episode_actions[agent.id] = action_dict
        timestep, arrivals = self.simulator.step()
        travel_times = {int(veh_id): {kc.TRAVEL_TIME : (timestep - self.episode_actions[int(veh_id)][kc.AGENT_START_TIME]) / 60.0} for veh_id in arrivals}
        for key in travel_times:    travel_times[key].update(self.episode_actions[key])
        return travel_times.values()
    
    #####################

    ##### DATA #####

    def get_free_flow_times(self):
        free_flow_times = self.simulator.get_free_flow_times()
        return free_flow_times

    #####################