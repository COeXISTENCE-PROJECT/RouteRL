import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from .learning_model import BaseLearningModel

class DQN(BaseLearningModel):
    def __init__(self, state_size, action_space_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.epsilon = 0.99
        self.epsilon_decay_rate = 0.01
        self.epsilon_min = 0.05
        self.memory = deque(maxlen=1000) ### WHA
        self.batch_size = 32
        self.learning_rate = 0.003
        num_hidden = 2
        widths = [32, 64, 32]

        
        self.q_network = Network(self.state_size, self.action_space_size, num_hidden, widths).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.loss = list()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def learn(self, state, action, reward):
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
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)


class Network(nn.Module):
    def __init__(self, state_size, action_space_size, num_hidden, widths):
        super(Network, self).__init__()
        assert len(widths) == (num_hidden + 1), "DQN widths and number of layers mismatch!"
        
        self.input_layer = nn.Linear(state_size, widths[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(widths[x], widths[x+1]) for x in range(num_hidden)])
        self.out_layer = nn.Linear(widths[-1], action_space_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.out_layer(x)
        return x