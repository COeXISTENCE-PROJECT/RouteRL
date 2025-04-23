import numpy as np
import random

from abc import ABC, abstractmethod

from routerl.keychain import Keychain as kc


class BaseLearningModel(ABC):
    """
    This is an abstract base class for the learning models used to model human learning and decision-making.\n
    Users can create their own learning models by inheriting from this class.
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
    """
    The Gawron learning model. This model is based on: `Gawron (1998) <https://kups.ub.uni-koeln.de/9257/>`_\n
    In summary, it iteratively shifts the cost expectations towards the received reward.\n
    For decision-making, calculates action utilities based on the ``beta`` parameter and cost expectations, and selects the action with the lowest utility.
    
    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    Attributes:
        beta (float): A parameter representing deviations in individual decision-making.
        alpha_zero (float): Agent's adaptation to new experiences.
        alpha_j (float): Weight for previous cost expectation (1 - ALPHA_ZERO).
        cost (np.ndarray): Agent's cost expectations for each option.
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
        """Selects an action based on the cost expectations.

        Args:
            state (Any): The current state of the environment (not used).
        Returns:
            action (int): The index of the selected action.
        """

        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        action =  utilities.index(min(utilities))
        return action   

    def learn(self, state, action, reward) -> None:
        """Updates the cost associated with the taken action based on the received reward.

        Args:
            state (string): The current state of the environment (not used).
            action (int): The action that was taken.
            reward (float): The reward received after taking the action.
        Returns:
            None
        """
        self.cost[action] = (self.alpha_j * self.cost[action]) + (self.alpha_zero * reward)


class Culo(BaseLearningModel):
    """
    The CUmulative LOgit learning model. This model is based on: `Li et al. (2024) <https://pubsonline.informs.org/doi/abs/10.1287/trsc.2023.0132/>`_.\n
    In summary, it updates its cost expectations by iteratively accumulating perceived rewards.\n
    For decision-making, calculates action utilities based on the ``beta`` parameter and cost expectations, and selects the action with the lowest utility.

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    Attributes:
        beta (float): A parameter representing deviations in individual decision-making.
        alpha_zero (float): Agent's adaptation to new experiences.
        alpha_j (float): Weight for previous cost expectation (constant = 1).
        cost (np.ndarray): Agent's cost expectations for each option.
    """

    def __init__(self, params, initial_knowledge):
        super().__init__()

        # Extract beta with randomness
        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = random.uniform(params[kc.BETA] - beta_randomness, params[kc.BETA] + beta_randomness)

        # Learning rate components
        self.alpha_zero = params[kc.ALPHA_ZERO]
        self.alpha_j = 1

        # Initialize cost array with initial knowledge
        self.cost = np.array(initial_knowledge, dtype=float)

    def act(self, state) -> int:
        """Selects an action based on the cost expectations.

        Args:
            state (Any): The current state of the environment (not used).
        Returns:
            action (int): The index of the selected action.
        """

        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        action =  utilities.index(min(utilities))
        return action   

    def learn(self, state, action, reward) -> None:
        """Updates the cost associated with the taken action based on the received reward.

        Args:
            state (Any): The current state of the environment (not used).
            action (int): The action that was taken.
            reward (float): The reward received after taking the action.
        Returns:
            None
        """

        self.cost[action] = (self.alpha_j * self.cost[action]) + (self.alpha_zero * reward)


class WeightedAverage(BaseLearningModel):
    """
    Weighted Average learning model. Theory based on: `Cascetta (2009) <https://link.springer.com/book/10.1007/978-0-387-75857-2/>`_.\n
    In summary, the model uses the reward and a weighted average of the past cost expectations to update the current cost expectation.\n
    For decision-making, calculates action utilities based on the ``beta`` parameter and cost expectations, and selects the action with the lowest utility.
    

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    Attributes:
        beta (float): A parameter representing deviations in individual decision-making.
        alpha_zero (float): Agent's adaptation to new experiences.
        alpha_j (float): Weight for previous cost expectation (1 - ALPHA_ZERO).
        remember (string): Memory size.
        cost (np.ndarray): Agent's cost expectations for each option.
        memory (list(list)): A list of lists containing the memory of each state.
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
        """Selects an action based on the cost expectations.

        Args:
            state (Any): The current state of the environment (not used).
        Returns:
            action (int): The index of the selected action.
        """

        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        action =  utilities.index(min(utilities))
        return action

    def learn(self, state, action, reward) -> None:
        """Updates the cost associated with the taken action based on the received reward.

        Args:
            state (Any): The current state of the environment (not used).
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
        """
        Creates a memory of previous cost expectations.

        Returns:
            None
        """

        for i in range(len(self.cost)):
            for _ in range(self.remember):
                self.memory[i].append(self.cost[i])
                
                
class Random(BaseLearningModel):
    """
    Random learning model. This model selects actions randomly without any learning or cost expectations.\n
    It is useful for testing and debugging purposes.

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    Attributes:
        cost (np.ndarray): Agent's cost expectations for each option.
    """

    def __init__(self, params, initial_knowledge):
        super().__init__()
        self.cost = np.array(initial_knowledge, dtype=float)
        
    def act(self, state) -> int:
        """Selects an action randomly.

        Args:
            state (Any): The current state of the environment (not used).
        Returns:
            action (int): The index of the selected action.
        """

        action = random.randint(0, len(self.cost) - 1)
        return action
    
    def learn(self, state, action, reward) -> None:
        """Does not learn or update any cost expectations.

        Args:
            state (Any): The current state of the environment (not used).
            action (int): The action that was taken.
            reward (float): The reward received after taking the action.
        Returns:
            None
        """
        pass
    
class AON(BaseLearningModel):
    """
    AON learning model. This model does not learn, but selects the action with the lowest cost expectation.\n
    It is useful for testing and debugging purposes.
    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    Attributes:
        cost (np.ndarray): Agent's cost expectations for each option.
    """
    def __init__(self, params, initial_knowledge):
        super().__init__()
        self.cost = np.array(initial_knowledge, dtype=float)
        
    def act(self, state) -> int:
        """Selects the action with the lowest cost expectation.

        Args:
            state (Any): The current state of the environment (not used).
        Returns:
            action (int): The index of the selected action.
        """

        action = int(np.argmax(self.cost))
        return action
    
    def learn(self, state, action, reward) -> None:
        """Does not learn or update any cost expectations.

        Args:
            state (Any): The current state of the environment (not used).
            action (int): The action that was taken.
            reward (float): The reward received after taking the action.
        Returns:
            None
        """
        pass