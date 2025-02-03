import numpy as np
import random

from abc import ABC, abstractmethod

from ..keychain import Keychain as kc



class BaseLearningModel(ABC):
    """
    This is an abstract base class for the learning models used to train the human and machine agents.

    """
    def __init__(self):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def learn(self, state, action, reward):
        pass



class Gawron(BaseLearningModel):
    def __init__(self, params, initial_knowledge):
        """
        Initializes the Gawron learning model.

        Parameters:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge or costs for actions.

        """

        super().__init__()

        # Extract beta with added randomness
        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = random.uniform(params[kc.BETA] - beta_randomness, params[kc.BETA] + beta_randomness)

        # Learning rate components
        self.alpha_zero = params[kc.ALPHA_ZERO]
        self.alpha_j = 1.0 - self.alpha_zero

        # Initialize cost array with initial knowledge
        self.cost = np.array(initial_knowledge, dtype=float)

    def act(self, state):
        """
        Selects an action based on the current state and cost.

        Parameters:
        state: The current state of the environment.

        Returns:
        int: The index of the selected action.
        """

        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        action =  utilities.index(min(utilities))
        return action   

    def learn(self, state, action, reward):
        """
        Updates the cost associated with the taken action based on the received reward.

        Parameters:
        state: The current state of the environment.
        action (int): The action that was taken.
        reward (float): The reward received after taking the action.

        """
        self.cost[action] = (self.alpha_j * self.cost[action]) + (self.alpha_zero * reward)


class Culo(BaseLearningModel):
    def __init__(self, params, initial_knowledge):
        """
        Initializes the Culo learning model.

        Parameters:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge or costs for actions.
        """
        super().__init__()

        # Extract beta with randomness
        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = random.uniform(params[kc.BETA] - beta_randomness, params[kc.BETA] + beta_randomness)

        # Learning rate components
        self.alpha_zero = 1
        self.alpha_j = params[kc.ALPHA_J]

        # Initialize cost array with initial knowledge
        self.cost = np.array(initial_knowledge, dtype=float)

    def act(self, state) -> int:
        """
        Selects an action based on the current state and cost.

        Parameters:
        state: The current state of the environment.

        Returns:
        int: The index of the selected action.
        """
        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        action =  utilities.index(min(utilities))
        return action   

    def learn(self, state, action, reward):
        """
        Updates the cost associated with the taken action based on the received reward.

        Parameters:
        state: The current state of the environment.
        action (int): The action that was taken.
        reward (float): The reward received after taking the action.
        """
        self.cost[action] = (self.alpha_j * self.cost[action]) + (self.alpha_zero * reward)


class WeightedAverage(BaseLearningModel):
    def __init__(self, params, initial_knowledge):
        super().__init__()
        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = random.uniform(params[kc.BETA] - beta_randomness, params[kc.BETA] + beta_randomness)
        self.alpha_zero = params[kc.ALPHA_ZERO]
        self.alpha_j = 1.0 - self.alpha_zero
        self.remember = params[kc.REMEMBER]
        self.cost = np.array(initial_knowledge, dtype=float)
        self.memory = [list() for _ in range(len(initial_knowledge))]
        self.create_memory()

    def act(self, state):
        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        action =  utilities.index(min(utilities))
        return action

    def learn(self, state, action, reward):
        c_hat = 0
        #if self.cost[action] != reward: # For next two lines
        # Drop the least relevant memory (end of list)
        del(self.memory[action][-1])
        # Insert the most recent expected cost at index 0
        self.memory[action].insert(0, self.cost[action])
            
        # Calculate the weights of the memory
        # The weights are proportional to its recency
        coeffs = [(self.remember - memory_idx) for memory_idx in range(self.remember)]
        # If remember=3, then coeffs = [3, 2, 1]. Now normalize the coeffs
        coeffs_normalized = [coeff / sum(coeffs) for coeff in coeffs]
        # Calculate the weighted average of the memory
        for memory_idx, coeff in enumerate(coeffs_normalized):
            c_hat += coeff * self.memory[action][memory_idx]
        # Update the cost expectation of the action
        self.cost[action] = (self.alpha_j * c_hat) + (self.alpha_zero * reward)
        
    def create_memory(self):
        for i in range(len(self.cost)):
            for _ in range(self.remember):
                self.memory[i].append(self.cost[i])