import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim

from abc import ABC, abstractmethod
from collections import deque

from keychain import Keychain as kc
from ..utilities import list_to_string



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



############## Human Learning ##############


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
        self.alpha_sigma = 1 - self.alpha_zero

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
        self.cost[action] = (self.alpha_sigma * self.cost[action]) + (self.alpha_zero * reward)


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
        self.alpha_sigma = params[kc.ALPHA_SIGMA]

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
        self.cost[action] = (self.alpha_sigma * self.cost[action]) + (self.alpha_zero * reward)


class WeightedAverage(BaseLearningModel):
    def __init__(self, params, initial_knowledge):
        super().__init__()
        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = random.uniform(params[kc.BETA] - beta_randomness, params[kc.BETA] + beta_randomness)
        self.alpha_zero = 0
        self.alpha_sigma = params[kc.ALPHA_SIGMA]
        self.cost = np.array(initial_knowledge, dtype=float)

    def act(self, state):
        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        action =  utilities.index(min(utilities))
        return action   

    #### Check with Onur to save previous reward
    def learn(self, state, action, reward):
        self.cost[action] = (self.alpha_sigma * self.cost[action]) + (self.alpha_zero * reward)



############## Machine Learning ##############



class QLearning(BaseLearningModel):
    def __init__(self, params, action_space_size):
        """
        Initializes the Q-learning model.

        Parameters:
        params (dict): A dictionary containing model parameters.
        action_space_size (int): The number of available actions in the environment.
        """
        super().__init__()

        # Learning parameters
        self.alpha = params[kc.ALPHA]
        self.epsilon = params[kc.EPSILON]
        self.epsilon_decay_rate = params[kc.EPSILON_DECAY_RATE]
        self.action_space_size = action_space_size

        # Initialize Q-table with zero values for each action
        self.sample_q_table_row = np.zeros((action_space_size))
        self.q_table = pd.DataFrame(columns=[kc.STATE, kc.Q_TABLE])


    def act(self, state):
        """
        Selects an action based on the current state using the epsilon-greedy strategy.

        Parameters:
        state (list): The current state of the environment.

        Returns:
        int: The index of the selected action.
        """
        state = list_to_string(state, separator="_")
        self.ensure_row_in_q_table(state)
        if np.random.rand() < self.epsilon:    # Explore
            return np.random.choice(self.action_space_size)
        else:    # Exploit
            q_values = self.q_table.loc[self.q_table[kc.STATE] == state, kc.Q_TABLE].item()
            return np.argmin(q_values)


    def learn(self, state, action, reward):
        """
        Updates the Q-values for the given state-action pair based on the reward.

        Parameters:
        state (list): The current state of the environment.
        action (int): The action that was taken.
        reward (float): The reward received after taking the action.
        """

        state = list_to_string(state, separator="_")
        q_row = self.q_table.loc[self.q_table[kc.STATE] == state, kc.Q_TABLE].item()
        prev_knowledge = q_row[action]
        q_row[action] = prev_knowledge + (self.alpha * (reward - prev_knowledge)) 
        self.decay_epsilon()


    def ensure_row_in_q_table(self, state):
        """
        Ensures the state is present in the Q-table. If not, a new row is added.

        Parameters:
        state_str (str): The string representation of the current state.
        """
        if not (state in self.q_table[kc.STATE].values):
            self.q_table.loc[len(self.q_table.index)] = {kc.STATE: state, kc.Q_TABLE: self.sample_q_table_row}
      
    def decay_epsilon(self):
        """
        Decays the epsilon value to reduce exploration over time.
        """

        self.epsilon *= self.epsilon_decay_rate



class DQN(BaseLearningModel):
    def __init__(self, params, state_size, action_space_size):
        """
        Initializes the Deep Q-learning Network (DQN).

        Parameters:
        params (dict): A dictionary containing model parameters.
        state_size (int): The dimensionality of the state space.
        action_space_size (int): The number of available actions.
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Environment and model parameters
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.epsilon = params[kc.EPSILON]
        self.epsilon_decay_rate = params[kc.EPSILON_DECAY_RATE]
        self.memory = deque(maxlen=params[kc.BUFFER_SIZE])
        self.batch_size = params[kc.BATCH_SIZE]
        self.learning_rate = params[kc.LEARNING_RATE]
        num_hidden = params[kc.NUM_HIDDEN]
        widths = params[kc.WIDTHS]

        # Initialize Q-network and optimizer
        self.q_network = Network(self.state_size, self.action_space_size, num_hidden, widths).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.loss = list()

    def act(self, state) -> int:
        """
        Selects an action based on the current state using an epsilon-greedy strategy.

        Parameters:
        state (array-like): The current state of the environment.

        Returns:
        int: The index of the selected action.
        """

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmin(q_values).item()

    def learn(self, state, action, reward):
        """
        Samples a batch from memory, computes the loss, and updates the Q-network.
        """
        self.memory.append((state, action, reward))
        if len(self.memory) < self.batch_size: return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards = zip(*batch)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)
        target_q_values = rewards_tensor

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.decay_epsilon()
        self.loss.append(loss.item())

    def decay_epsilon(self):
        """
        Decays the epsilon value to reduce exploration over time.
        """
        self.epsilon *= self.epsilon_decay_rate





class Network(nn.Module):
    def __init__(self, state_size, action_space_size, num_hidden, widths):
        """
        Initializes the neural network for the Deep Q-Learning Network (DQN).

        Parameters:
        state_size (int): The dimensionality of the input state space.
        action_space_size (int): The number of available actions (output size).
        num_hidden (int): The number of hidden layers in the network.
        widths (list): A list defining the width of each hidden layer and the input layer.
                       The length of the list must match num_hidden + 1.
        """
        super(Network, self).__init__()
        assert len(widths) == (num_hidden + 1), "DQN widths and number of layers mismatch!"
        
        self.input_layer = nn.Linear(state_size, widths[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(widths[x], widths[x+1]) for x in range(num_hidden)])
        self.out_layer = nn.Linear(widths[-1], action_space_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        x (Tensor): The input tensor representing the state.

        Returns:
        Tensor: The output tensor representing Q-values for each action.
        """
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.out_layer(x)
        return x