import numpy as np
import random

from abc import ABC, abstractmethod

from routerl.keychain import Keychain as kc


class BaseLearningModel(ABC):
    """This is an abstract base class for the learning models used to train the human and machine agents.

    Methods:
        act: selects an action based on the current state and cost.
        learn: trains the model on a batch of states and actions.
    """

    def __init__(self):
        pass

    @abstractmethod
    def act(self, state) -> None:
        """Method to select an action based on the current state and cost.

        Returns:
            None
        """
        pass

    @abstractmethod
    def learn(self, state, action, reward) -> None:
        """Method to learn the model based on the current state and cost.

        Arguments:
            state (Any): The current state of the environment.
            action (Any): The action to take.
            reward (Any): The reward received from the environment.
        Returns:
            None
        """

        pass


class Gawron(BaseLearningModel):
    """The Gawron learning model.

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge or costs for actions.
    Attributes:
        beta (float): random number between BETA-beta_randomness and BETA+beta_randomness.
        alpha_zero (float): ALPHA_ZERO.
        alpha_j (float): ALPHA_J = 1 - ALPHA_ZERO.
        cost (np.ndarray): cost array.
    Methods:
        act: selects an action based on the current state and cost.
        learn: trains the model on a batch of states and actions.
    """

    def __init__(self, params, initial_knowledge):
        super().__init__()

        # Extract beta with added randomness
        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = random.uniform(params[kc.BETA] - beta_randomness, params[kc.BETA] + beta_randomness)

        # Learning rate components
        self.alpha_zero = params[kc.ALPHA_ZERO]
        self.alpha_j = 1.0 - self.alpha_zero

        # Initialize cost array with initial knowledge
        self.cost = np.array(initial_knowledge, dtype=float)

    def act(self, state) -> int:
        """Selects an action based on the current state and cost.

        Args:
            state (string): The current state of the environment.
        Returns:
            action (int): The index of the selected action.
        """

        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        # fixme rename action
        action =  utilities.index(min(utilities))
        return action   

    def learn(self, state, action, reward) -> None:
        """Updates the cost associated with the taken action based on the received reward.

        Args:
            state (string): The current state of the environment.
            action (int): The action that was taken.
            reward (float): The reward received after taking the action.
        Returns:
            None
        """

        self.cost[action] = (self.alpha_j * self.cost[action]) + (self.alpha_zero * reward)


class Culo(BaseLearningModel):
    """The Culo learning model.

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge or costs for actions.
    Attributes:
        beta (float): random number between BETA-beta_randomness and BETA+beta_randomness.
        alpha_zero (float): ALPHA_ZERO.
        alpha_j (float): ALPHA_J = 1 - ALPHA_ZERO.
        cost (np.ndarray): cost array.
    Methods:
        act: selects an action based on the current state and cost.
        learn: trains the model on a batch of states and actions.
    """

    def __init__(self, params, initial_knowledge):
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
        """Selects an action based on the current state and cost.

        Args:
            state (string): The current state of the environment.
        Returns:
            action (int): The index of the selected action.
        """

        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        action =  utilities.index(min(utilities))
        return action   

    def learn(self, state, action, reward) -> None:
        """Updates the cost associated with the taken action based on the received reward.

        Args:
            state (string): The current state of the environment.
            action (int): The action that was taken.
            reward (float): The reward received after taking the action.
        Returns:
            None
        """

        self.cost[action] = (self.alpha_j * self.cost[action]) + (self.alpha_zero * reward)


class WeightedAverage(BaseLearningModel):
    """Weighted Average model.

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge or costs for actions.
    Attributes:
        beta (float): random number between BETA-beta_randomness and BETA+beta_randomness.
        alpha_zero (float): ALPHA_ZERO.
        alpha_j (float): ALPHA_J = 1 - ALPHA_ZERO.
        remember (string): REMEMBER
        cost (np.ndarray): cost array.
        memory (list(list)): A list of lists containing the memory of each state.
    Methods:
        act: selects an action based on the current state and cost.
        learn: trains the model on a batch of states and actions.
        create_memory: creates 2 dim memory table.
    """

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

    def act(self, state) -> int:
        """Selects an action based on the current state and cost.

        Args:
            state (string): The current state of the environment.
        Returns:
            action (int): The index of the selected action.
        """

        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        action =  utilities.index(min(utilities))
        return action

    def learn(self, state, action, reward) -> None:
        """Updates the cost associated with the taken action based on the received reward.

        Args:
            state (string): The current state of the environment.
            action (int): The action that was taken.
            reward (float): The reward received after taking the action.
        Returns:
            None
        """

        # Drop the least relevant memory (end of list)
        del(self.memory[action][-1])
        # Insert the most recent expected cost at index 0
        self.memory[action].insert(0, self.cost[action])
        
        # Calculate the weights of the memory
        # The weights are proportional to item recency
        alpha_j_weights = [self.alpha_j / (memory_idx + 1) for memory_idx in range(self.remember)]
        # If remember=3 alpha_j=.5, then alpha_j_weights = [.5/1, .5/2, .5/3]. Now normalize alpha_j_weights.
        alpha_j_normalized = [a_j / sum(alpha_j_weights) for a_j in alpha_j_weights]
        
        # Calculate the weighted average of the memory
        c_hat = 0
        for memory_idx, a_j in enumerate(alpha_j_normalized):
            c_hat += a_j * self.memory[action][memory_idx]
            
        # Update the cost expectation of the action
        self.cost[action] = c_hat + (self.alpha_zero * reward)
        

    def create_memory(self) -> None:
        """Creates 2 dim memory table.

        Returns:
            None
        """

        for i in range(len(self.cost)):
            for _ in range(self.remember):
                self.memory[i].append(self.cost[i])