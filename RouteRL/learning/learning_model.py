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
        raise NotImplementedError("WeightedAverage learning model is not implemented yet.")
        super().__init__()
        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = random.uniform(params[kc.BETA] - beta_randomness, params[kc.BETA] + beta_randomness)
        self.alpha_zero = 0
        self.alpha_j = params[kc.ALPHA_J]
        self.cost = np.array(initial_knowledge, dtype=float)

    def act(self, state):
        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        action =  utilities.index(min(utilities))
        return action   

    def learn(self, state, action, reward):
        self.cost[action] = (self.alpha_j * self.cost[action]) + (self.alpha_zero * reward)