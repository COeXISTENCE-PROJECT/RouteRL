from abc import ABC, abstractmethod

from keychain import Keychain as kc
from learning import Gawron, QLearning, DQN



class Agent(ABC):
    """
    This is an abstract class for agents, to be inherited by specific type of agent classes
    It is not to be instantiated, but to provide a blueprint for all types of agents
    """
    def __init__(self, id, kind, start_time, origin, destination):
        self.id = id
        self.kind = kind
        self.start_time = start_time
        self.origin = origin
        self.destination = destination

    @abstractmethod
    def act(self, observation):  
        # Pick action according to your knowledge, or randomly
        pass

    @abstractmethod
    def learn(self, action, observation):
        # Pass the applied action and reward once the episode ends, and it will remember the consequences
        pass

    @abstractmethod
    def get_state(self, observation):
        # Return the state of the agent, given the observation
        pass



class HumanAgent(Agent):
    def __init__(self, id, start_time, origin, destination, params, initial_knowledge, mutate_to=None, mutate_type=None):
        kind = kc.TYPE_HUMAN
        super().__init__(id, kind, start_time, origin, destination)
        self.mutate_to = mutate_to
        self.mutate_type = mutate_type
        self.model = Gawron(params, initial_knowledge)

    def act(self, observation):  
        return self.model.act(observation)  

    def learn(self, action, reward):
        self.model.learn(action, reward)

    def get_state(self, observation):
        pass
    
    def mutate(self):
        return self.mutate_to
    


class MachineAgent(Agent):
    def __init__(self, id, start_time, origin, destination, params, action_space_size):
        kind = kc.TYPE_MACHINE
        super().__init__(id, kind, start_time, origin, destination)
        self.model = QLearning(params, action_space_size)
        self.last_state = None

    def act(self, observation):
        state = self.get_state(observation)
        self.last_state = state
        return self.model.act(state)

    def learn(self, action, reward):
        self.model.learn(self.last_state, action, reward)

    def get_state(self, observation):
        return observation



class DisruptiveMachineAgent(Agent):
    def __init__(self, id, start_time, origin, destination, params, action_space_size):
        kind = kc.TYPE_MACHINE_2
        super().__init__(id, kind, start_time, origin, destination)
        self.action_space_size = action_space_size
        self.state_size = action_space_size
        self.model = DQN(params, self.state_size, self.action_space_size)
        self.last_state = None

    def act(self, observation):
        state = self.get_state(observation)
        self.last_state = state
        return self.model.act(state)

    def learn(self, action, reward):
        self.model.learn(self.last_state, action, reward)

    def get_state(self, observation):
        state = [0] * self.action_space_size
        min_start_time = self.start_time - 10
        if not observation.empty:
            for _, row in observation.iterrows():
                if row[kc.AGENT_START_TIME] < min_start_time:
                    continue
                action = row[kc.ACTION]
                start_time = row[kc.AGENT_START_TIME]
                state[action] += start_time
            if sum(state):
                state = [x / sum(state) for x in state]
        return state