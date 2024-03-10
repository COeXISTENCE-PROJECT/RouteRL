import numpy as np
import pandas as pd
import random

from abc import ABC, abstractmethod

from keychain import Keychain as kc
from utilities import list_to_string


class Agent(ABC):

    """
    This is an abstract class for agents, to be inherited by specific type of agent classes
    It is not to be instantiated, but to provide a blueprint for all types of agents
    """
    
    def __init__(self, id, start_time, origin, destination):
        self.id = id

        self.start_time = start_time
        self.origin = origin
        self.destination = destination

    @abstractmethod
    def act(self, observation):  
        # Pick action according to your knowledge, or randomly
        pass

    @abstractmethod
    def get_state(self, observation):
        # Return the state of the agent, given the observation
        pass

    @abstractmethod
    def learn(self, action, observation):
        # Pass the applied action and reward once the episode ends, and it will remember the consequences
        pass

    @abstractmethod
    def get_reward(self, observation):
        # Return the reward of the agent, given the observation
        pass




class HumanAgent(Agent):

    def __init__(self, id, start_time, origin, destination, params, initial_knowledge, mutate_to=None, mutate_type=None):
        super().__init__(id, start_time, origin, destination)

        self.kind = kc.TYPE_HUMAN
        self.mutate_to = mutate_to
        self.mutate_type = mutate_type

        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = random.uniform(params[kc.BETA] - beta_randomness, params[kc.BETA] + beta_randomness)
        self.alpha = params[kc.ALPHA]

        self.cost = np.array(initial_knowledge, dtype=float)


    def act(self, observation):  
        state = self.get_state(observation)
        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        prob_dist = [self.calculate_prob(utilities, j) for j in range(len(self.cost))]
        action = np.random.choice(list(range(len(self.cost))), p=prob_dist) 
        return action


    def get_state(self, observation):
        return list_to_string(observation, separator="_")        


    def learn(self, action, observation):
        reward = self.get_reward(observation)
        self.cost[action] = ((1-self.alpha) * self.cost[action]) + (self.alpha * reward)


    def get_reward(self, observation):
        reward = observation.loc[observation[kc.AGENT_ID] == self.id, kc.TRAVEL_TIME].item()
        return reward


    def calculate_prob(self, utilities, n):
        prob = utilities[n] / sum(utilities)
        return prob
    

    def mutate(self):
        self.mutate_to.receive_initial_knowledge(self.cost)
        return self.mutate_to
    



class MachineAgent(Agent):

    def __init__(self, id, start_time, origin, destination, params, action_space_size):
        super().__init__(id, start_time, origin, destination)

        self.kind = kc.TYPE_MACHINE

        min_alpha, max_alpha = params[kc.MIN_ALPHA], params[kc.MAX_ALPHA]
        min_epsilon, max_epsilon = params[kc.MIN_EPSILON], params[kc.MAX_EPSILON]
        min_eps_decay, max_eps_decay = params[kc.MIN_EPS_DECAY], params[kc.MAX_EPS_DECAY]

        self.epsilon = random.uniform(min_epsilon, max_epsilon)
        self.epsilon_decay_rate = random.uniform(min_eps_decay, max_eps_decay)
        self.alpha = random.uniform(min_alpha, max_alpha)
        self.gamma = params[kc.GAMMA]

        self.action_space_size = action_space_size
        # Q-table assumes only one state, otherwise should be np.zeros((num_states, action_space_size))
        self.sample_q_table_row = np.zeros((action_space_size))
        self.q_table = pd.DataFrame(columns=[kc.STATE, kc.Q_TABLE])

        self.last_state = None


    def act(self, observation):
        state = self.get_state(observation)
        self.ensure_row_in_q_table(state)
        self.last_state = state
        if np.random.rand() < self.epsilon:    # Explore
            return np.random.choice(self.action_space_size)
        else:    # Exploit
            table = self.q_table.loc[self.q_table[kc.STATE] == state, kc.Q_TABLE].item()
            return np.argmin(table)
        

    def get_state(self, observation):
        return list_to_string(observation, separator="_")


    def ensure_row_in_q_table(self, state):
        if not (state in self.q_table[kc.STATE].values):
            self.q_table.loc[len(self.q_table.index)] = {kc.STATE: state, kc.Q_TABLE: self.sample_q_table_row}
                

    def learn(self, action, observation):
        reward = self.get_reward(observation)
        table = self.q_table.loc[self.q_table[kc.STATE] == self.last_state, kc.Q_TABLE].item()
        prev_knowledge = table[action]
        table[action] = prev_knowledge + (self.alpha * (reward - prev_knowledge))
        self.decay_epsilon()
        

    def get_reward(self, observation):
        reward = observation.loc[observation[kc.AGENT_ID] == self.id, kc.TRAVEL_TIME].item()
        return reward


    def decay_epsilon(self):    # Slowly become deterministic
        self.epsilon *= self.epsilon_decay_rate

    
    def receive_initial_knowledge(self, initial_knowledge):
        self.sample_q_table_row = initial_knowledge




class MaliciousMachineAgent(Agent):

    def __init__(self, id, start_time, origin, destination, params, action_space_size):
        super().__init__(id, start_time, origin, destination)

        self.kind = kc.TYPE_MACHINE_2

        min_alpha, max_alpha = params[kc.MIN_ALPHA], params[kc.MAX_ALPHA]
        min_epsilon, max_epsilon = params[kc.MIN_EPSILON], params[kc.MAX_EPSILON]
        min_eps_decay, max_eps_decay = params[kc.MIN_EPS_DECAY], params[kc.MAX_EPS_DECAY]

        self.epsilon = random.uniform(min_epsilon, max_epsilon)
        self.epsilon_decay_rate = random.uniform(min_eps_decay, max_eps_decay)
        self.alpha = random.uniform(min_alpha, max_alpha)
        self.gamma = params[kc.GAMMA]

        self.action_space_size = action_space_size
        # Q-table assumes only one state, otherwise should be np.zeros((num_states, action_space_size))
        self.sample_q_table_row = np.zeros((action_space_size))
        self.q_table = pd.DataFrame(columns=[kc.STATE, kc.Q_TABLE])

        self.last_state = None


    def act(self, observation):
        state = self.get_state(observation)
        self.ensure_row_in_q_table(state)
        self.last_state = state
        if np.random.rand() < self.epsilon:    # Explore
            return np.random.choice(self.action_space_size)
        else:    # Exploit
            table = self.q_table.loc[self.q_table[kc.STATE] == state, kc.Q_TABLE].item()
            return np.argmax(table)
        

    def get_state(self, observation):
        return list_to_string(observation, separator="_")


    def ensure_row_in_q_table(self, state):
        if not (state in self.q_table[kc.STATE].values):
            self.q_table.loc[len(self.q_table.index)] = {kc.STATE: state, kc.Q_TABLE: self.sample_q_table_row}
                

    def learn(self, action, observation):
        reward = self.get_reward(observation)
        table = self.q_table.loc[self.q_table[kc.STATE] == self.last_state, kc.Q_TABLE].item()
        prev_knowledge = table[action]
        table[action] = prev_knowledge + (self.alpha * (reward - prev_knowledge))
        self.decay_epsilon()
        

    def get_reward(self, observation):
        reward = observation.loc[observation[kc.AGENT_ID] == self.id, kc.TRAVEL_TIME].item()
        return reward


    def decay_epsilon(self):    # Slowly become deterministic
        self.epsilon *= self.epsilon_decay_rate

    
    def receive_initial_knowledge(self, initial_knowledge):
        self.sample_q_table_row = initial_knowledge