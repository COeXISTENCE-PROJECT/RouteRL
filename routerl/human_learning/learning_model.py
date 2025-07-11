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
        alpha (float): Agent's adaptation to new experiences.
        cost (np.ndarray): Agent's cost expectations for each option.
    """

    def __init__(self, params, initial_knowledge):
        super().__init__()

        # Extract beta with added randomness
        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = random.uniform(params[kc.BETA] - beta_randomness, params[kc.BETA] + beta_randomness)

        # Learning rate
        self.alpha = params[kc.ALPHA]

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
        prob_dist = [self.calculate_prob(utilities, idx) for idx in range(len(self.cost))]
        action = np.random.choice(list(range(len(self.cost))), p=prob_dist) 
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
        self.cost[action] = ((1.0 - self.alpha) * self.cost[action]) + (self.alpha * reward)

    def calculate_prob(self, utilities, n):
        prob = utilities[n] / sum(utilities)
        return prob


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
        alpha (float): Agent's adaptation to new experiences.
        cost (np.ndarray): Agent's cost expectations for each option.
    """

    def __init__(self, params, initial_knowledge):
        super().__init__()

        # Extract beta with randomness
        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = random.uniform(params[kc.BETA] - beta_randomness, params[kc.BETA] + beta_randomness)

        # Learning rate components
        self.alpha = params[kc.ALPHA]

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
        prob_dist = [self.calculate_prob(utilities, idx) for idx in range(len(self.cost))]
        action = np.random.choice(list(range(len(self.cost))), p=prob_dist) 
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

        self.cost[action] = self.cost[action] + (self.alpha * reward)

    def calculate_prob(self, utilities, n):
        prob = utilities[n] / sum(utilities)
        return prob


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
        alpha (float): Agent's adaptation to new experiences.
        alpha_past (float): Weight for previous cost expectation (1 - ALPHA).
        remember (string): Memory size.
        cost (np.ndarray): Agent's cost expectations for each option.
        memory (list(list)): A list of lists containing the memory of each state.
    """

    def __init__(self, params, initial_knowledge):
        super().__init__()
        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = random.uniform(params[kc.BETA] - beta_randomness, params[kc.BETA] + beta_randomness)
        self.alpha = params[kc.ALPHA]
        self.alpha_past = 1.0 - self.alpha
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
        prob_dist = [self.calculate_prob(utilities, idx) for idx in range(len(self.cost))]
        action = np.random.choice(list(range(len(self.cost))), p=prob_dist) 
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
        alpha_past_weights = [self.alpha_past / (memory_idx + 1) for memory_idx in range(self.remember)]
        # If remember=3 alpha_past=.5, then alpha_past_weights = [.5/1, .5/2, .5/3]. Now normalize alpha_past_weights.
        alpha_past_normalized = [a_j / sum(alpha_past_weights) for a_j in alpha_past_weights]
        
        # Calculate the weighted average of the memory
        c_hat = 0
        for memory_idx, a_j in enumerate(alpha_past_normalized):
            c_hat += a_j * self.memory[action][memory_idx]
            
        # Update the cost expectation of the action
        self.cost[action] = c_hat + (self.alpha * reward)
        

    def create_memory(self) -> None:
        """
        Creates a memory of previous cost expectations.

        Returns:
            None
        """

        for i in range(len(self.cost)):
            for _ in range(self.remember):
                self.memory[i].append(self.cost[i])
                
    def calculate_prob(self, utilities, n):
        prob = utilities[n] / sum(utilities)
        return prob
                
                
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