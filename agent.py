from abc import ABC, abstractmethod
from keychain import Keychain as kc

import numpy as np
import random


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
    def act(self, state):  
        # Pick action according to your knowledge, or randomly
        pass

    @abstractmethod
    def learn(self, action, reward, state, next_state):
        # Pass the applied action and reward once the episode ends, and it will remember the consequences
        pass




class HumanAgent(Agent):

    def __init__(self, id, start_time, origin, destination, params, initial_knowledge):
        super().__init__(id, start_time, origin, destination)

        learning_params = params[kc.HUMAN_AGENT_PARAMETERS]
        self.beta = learning_params[kc.BETA]
        self.alpha = learning_params[kc.ALPHA]

        self.cost = initial_knowledge


    def act(self, state):  
        """ 
        the implemented dummy logit model for route choice, make it more generate, calculate in graph levelbookd
        """
        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        prob_dist = [self.calculate_prob(utilities, j) for j in range(len(self.cost))]
        action = np.random.choice(list(range(len(self.cost))), p=prob_dist)    
        return action        


    def learn(self, action, reward, state, next_state):
        self.cost[action] = (1-self.alpha) * self.cost[action] + self.alpha * reward


    def calculate_prob(self, utilities, n):
        prob = utilities[n] / sum(utilities)
        return prob
    



class MachineAgent(Agent):

    def __init__(self, id, start_time, origin, destination, params, action_space_size):
        super().__init__(id, start_time, origin, destination)

        learning_params = params[kc.MACHINE_AGENT_PARAMETERS]

        min_alpha, max_alpha = learning_params[kc.MIN_ALPHA], learning_params[kc.MAX_ALPHA]
        min_epsilon, max_epsilon = learning_params[kc.MIN_EPSILON], learning_params[kc.MAX_EPSILON]
        min_eps_decay, max_eps_decay = learning_params[kc.MIN_EPS_DECAY], learning_params[kc.MAX_EPS_DECAY]

        self.epsilon = random.uniform(min_epsilon, max_epsilon)
        self.epsilon_decay_rate = random.uniform(min_eps_decay, max_eps_decay)
        self.alpha = random.uniform(min_alpha, max_alpha)
        self.gamma = learning_params[kc.GAMMA]

        self.action_space_size = action_space_size
        # Q-table assumes only one state, otherwise should be np.zeros((num_states, action_space_size))
        self.q_table = np.zeros((action_space_size))
        #self.q_table = initial_knowledge


    def act(self, state):
        if np.random.rand() < self.epsilon:    # Explore
            return np.random.choice(self.action_space_size)
        else:    # Exploit
            return np.argmax(self.q_table)
                

    def learn(self, action, reward, state, next_state):
        prev_knowledge = self.q_table[action]
        self.q_table[action] = prev_knowledge + (self.alpha * (reward + (self.gamma * np.argmax(self.q_table)) - prev_knowledge))
        self.decay_epsilon()


    def decay_epsilon(self):    # Slowly become deterministic
        self.epsilon *= self.epsilon_decay_rate