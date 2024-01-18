from abc import ABC, abstractmethod
import numpy as np
import random


class Agent(ABC):

    """
    This is an abstract class for agents, to be inherited by specific type of agent classes
    It is not to be instantiated, but to provide a blueprint for all types of agents
    """
    
    # We assume no state space for now
    def __init__(self, id, start_time, origin, destination, action_space_size):
        self.id = id

        self.start_time = start_time
        self.origin = origin
        self.destination = destination

        self.action_space_size = action_space_size

    @abstractmethod
    def learn(self):    # Pass the applied action and reward once the episode ends, and it will remember the consequences
        pass

    @abstractmethod
    def pick_action(self):  # Pick action according to your knowledge, or randomly
        pass



class HumanAgent(Agent):
    def __init__(self, id, start_time, origin, destination, action_space_size):
        super().__init__(id, start_time, origin, destination, action_space_size)

    def learn(self):
        # Implement Garwon learning model
        pass

    def pick_action(self):
        # Implement decision making for humans
        pass



class MachineAgent(Agent):
    def __init__(self, id, start_time, origin, destination, action_space_size, learning_params):
        super().__init__(id, start_time, origin, destination, action_space_size)

        min_alpha, max_alpha = learning_params["min_alpha"], learning_params["max_alpha"]
        min_epsilon, max_epsilon = learning_params["min_epsilon"], learning_params["max_epsilon"]
        min_eps_decay, max_eps_decay = learning_params["min_eps_decay"], learning_params["max_eps_decay"]

        self.epsilon = random.uniform(min_epsilon, max_epsilon)
        self.epsilon_decay_rate = random.uniform(min_eps_decay, max_eps_decay)
        self.alpha = random.uniform(min_alpha, max_alpha)

        # Q-table assumes only one state, otherwise should be np.zeros((num_states, action_space_size)), also edit the rest of the class accordingly
        self.q_table = np.zeros((action_space_size))    
        

    def learn(self, action, reward, state, next_state):
        prev_knowledge = self.q_table[action]
        self.q_table[action] = prev_knowledge + (self.alpha * (reward - prev_knowledge))
        self.decay_epsilon()


    def pick_action(self, state):
        if np.random.rand() < self.epsilon:    # Explore
            return np.random.choice(self.action_space_size)
        else:    # Exploit
            return np.argmax(self.q_table)


    def decay_epsilon(self):    # Slowly become deterministic
        self.epsilon *= self.epsilon_decay_rate