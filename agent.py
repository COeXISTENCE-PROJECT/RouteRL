from abc import ABC, abstractmethod

from keychain import Keychain as kc
from learning import Gawron, QLearning, DQN



class BaseAgent(ABC):
    """
    This is an abstract class for agents, to be inherited by specific type of agent classes
    It is not to be instantiated, but to provide a blueprint for all types of agents
    """
    def __init__(self, id, kind, start_time, origin, destination, behavior, learning_phases):
        self.id = id
        self.kind = kind
        self.start_time = start_time
        self.origin = origin
        self.destination = destination
        self.behavior = behavior
        self.learning_phases = learning_phases

    @property
    @abstractmethod
    def is_learning(self):
        # Return True if the agent is in a learning phase, False otherwise
        pass

    @is_learning.setter
    @abstractmethod
    def is_learning(self, phase):
        # Set the learning state of the agent
        pass

    @property
    @abstractmethod
    def last_reward(self):
        # Return the last reward of the agent
        pass
    
    @last_reward.setter
    @abstractmethod
    def last_reward(self, reward):
        # Set the last reward of the agent
        pass

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

    @abstractmethod
    def get_reward(self, observation):
        # Derive the reward of the agent, given the observation
        pass



class HumanAgent(BaseAgent):
    def __init__(self, id, start_time, origin, destination, params, initial_knowledge, mutate_to=None):
        kind = kc.TYPE_HUMAN
        behavior = kc.SELFISH
        learning_phases = params[kc.LEARNING_PHASES]
        super().__init__(id, kind, start_time, origin, destination, behavior, learning_phases)
        self.mutate_to = mutate_to
        self.model = Gawron(params, initial_knowledge) 
        self.is_learning = -1
        self.last_reward = None

    @property
    def is_learning(self):
        return self._is_learning

    @is_learning.setter
    def is_learning(self, phase):
        if phase in self.learning_phases:
            self._is_learning = True
        else:
            self._is_learning = False

    @property
    def last_reward(self):
        return self._last_reward
    
    @last_reward.setter
    def last_reward(self, reward):
        self._last_reward = reward

    @property
    def mutate_type(self):
        return getattr(self.mutate_to, 'kind', None)
    
    @property
    def mutate_phase(self):
        return getattr(self.mutate_to, 'appearance_phase', None)

    def act(self, observation):  
        return self.model.act(observation)  

    def learn(self, action, observation):
        if self.is_learning:
            reward = self.get_reward(observation)
            self.last_reward = reward
            self.model.learn(None, action, reward)

    def get_state(self, observation):
        return None

    def get_reward(self, observation):
        own_tt = observation.loc[observation[kc.AGENT_ID] == self.id, kc.TRAVEL_TIME].item()
        return own_tt
    
    def mutate(self):
        return self.mutate_to
    


class MachineAgent(BaseAgent):
    def __init__(self, id, start_time, origin, destination, params, action_space_size):
        kind = kc.TYPE_MACHINE
        behavior = params[kc.BEHAVIOR]
        learning_phases = params[kc.LEARNING_PHASES]
        super().__init__(id, kind, start_time, origin, destination, behavior, learning_phases)
        self.appearance_phase = params[kc.APPEARANCE_PHASE]
        self.action_space_size = action_space_size
        self.state_size = action_space_size
        self.model = DQN(params, self.state_size, self.action_space_size)
        self.is_learning = -1
        self.last_reward = None

    @property
    def is_learning(self):
        return self._is_learning

    @is_learning.setter
    def is_learning(self, phase):
        if phase in self.learning_phases:
            self._is_learning = True
        else:
            self._is_learning = False

    @property
    def last_reward(self):
        return self._last_reward
    
    @last_reward.setter
    def last_reward(self, reward):
        self._last_reward = reward

    def act(self, observation):
        state = self.get_state(observation)
        self.last_state = state
        return self.model.act(state)

    def learn(self, action, observation):
        if self.is_learning:
            reward = self.get_reward(observation)
            self.last_reward = reward
            self.model.learn(self.last_state, action, reward)

    def get_state(self, observation):
        state = [0] * self.state_size
        min_start_time = self.start_time - 10
        observation = observation.loc[(observation[kc.AGENT_ORIGIN] == self.origin) & (observation[kc.AGENT_DESTINATION] == self.destination)]
        prior_agents = observation.loc[observation[kc.AGENT_START_TIME] >= min_start_time]
        if not prior_agents.empty:
            for _, row in prior_agents.iterrows():
                action = row[kc.ACTION]
                start_time = row[kc.AGENT_START_TIME]
                warmth = start_time - min_start_time
                state[action] += warmth
        return state
    
    def get_reward(self, observation):
        min_start_time, max_start_time = self.start_time - 10, self.start_time + 10
        observation = observation.loc[(observation[kc.AGENT_ORIGIN] == self.origin) & (observation[kc.AGENT_DESTINATION] == self.destination)]
        vicinity_obs = observation.loc[(observation[kc.AGENT_START_TIME] >= min_start_time) & (observation[kc.AGENT_START_TIME] <= max_start_time)]

        group_obs = vicinity_obs.loc[vicinity_obs[kc.AGENT_KIND] == self.kind, kc.TRAVEL_TIME]
        others_obs = vicinity_obs.loc[vicinity_obs[kc.AGENT_KIND] != self.kind, kc.TRAVEL_TIME]

        own_tt = observation.loc[observation[kc.AGENT_ID] == self.id, kc.TRAVEL_TIME].item()
        group_tt = group_obs.mean() if not group_obs.empty else 0
        others_tt = others_obs.mean() if not others_obs.empty else 0
        all_tt = vicinity_obs[kc.TRAVEL_TIME].mean()
        
        a, b, c, d = self.reward_coefs()
        reward = a * own_tt + b * group_tt + c * others_tt + d * all_tt
        return reward
    
    def reward_coefs(self):
        a, b, c, d = 0, 0, 0, 0
        if self.behavior == kc.SELFISH:
            a, b, c, d = 1, 0, 0, 0
        elif self.behavior == kc.DISRUPTIVE:
            a, b, c, d = 1, 0, -1, 0
        elif self.behavior == kc.SOCIAL:
            a, b, c, d = 0.5, 0, 0, 0.5
        elif self.behavior == kc.ALTURISTIC:
            a, b, c, d = 0, 0, 0, 1
        return a, b, c, d
