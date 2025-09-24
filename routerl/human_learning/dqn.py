import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from .learning_model import BaseLearningModel

class DQN(BaseLearningModel):
    def __init__(self, state_size,
                action_space_size,
                epsilon=0.99,
                epsilon_decay_rate=0.01,
                epsilon_min=0.05,
                memory_size=1000,
                batch_size=32,
                learning_rate=0.003,
                widths=(32, 64, 32),
                device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.widths = widths

        self.q_network = Network(self.state_size, self.action_space_size, self.widths).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.loss = list()

    def train(self):
        """Set the model to training mode"""
        self.q_network.train()
        return self

    def eval(self):
        """Set the model to evaluation mode"""
        self.q_network.eval()
        return self

    def act(self, state):
        if self.q_network.training and np.random.rand() < self.epsilon:
            action = torch.randint(0, self.action_space_size, (1,)).item()
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
            
            self.last_state = state
            self.last_action = action

            return action

    def learn(self, state, action, reward):
        #print("I am inside the learn function\n\n")
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
        self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.epsilon_min)

class Network(nn.Module):
    def __init__(self, state_size, action_space_size, widths):
        super(Network, self).__init__()
        
        self.input_layer = nn.Linear(state_size, widths[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(widths[x], widths[x+1]) for x in range(len(widths) - 1)])
        self.out_layer = nn.Linear(widths[-1], action_space_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.out_layer(x)
        return x