import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from .learning_model import BaseLearningModel

class MAPPO(BaseLearningModel):
    def __init__(self, state_size, action_space_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_space_size = action_space_size
        # Hyperparameters
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.learning_rate = 0.0003
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.batch_size = 64
        self.memory = deque(maxlen=5000)
        # Networks
        self.actor = PolicyNetwork(self.state_size, self.action_space_size).to(self.device)
        self.critic = ValueNetwork(self.state_size).to(self.device)
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.loss_actor = []
        self.loss_critic = []

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.actor(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def store(self, state, action, reward, log_prob, next_state, done):
        self.memory.append((state, action, reward, log_prob, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, old_log_probs, next_states, dones = zip(*batch)
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        # Compute targets
        with torch.no_grad():
            target_values = rewards_tensor + self.gamma * self.critic(next_states_tensor) * (1 - dones_tensor)
        # Compute current values
        current_values = self.critic(states_tensor)
        value_loss = nn.MSELoss()(current_values, target_values)
        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        self.loss_critic.append(value_loss.item())
        # Compute advantages
        advantages = (target_values - current_values).detach()
        # Compute new log probs
        logits = self.actor(states_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions_tensor.squeeze(1)).unsqueeze(1)
        # Policy loss with clipping
        ratios = torch.exp(new_log_probs - old_log_probs_tensor)
        clipped_ratios = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
        # Add entropy bonus
        entropy = dist.entropy().mean()
        total_policy_loss = policy_loss - self.entropy_coef * entropy
        # Update actor
        self.actor_optimizer.zero_grad()
        total_policy_loss.backward()
        self.actor_optimizer.step()
        self.loss_actor.append(total_policy_loss.item())

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_space_size):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size)
        )
    def forward(self, x):
        return self.net(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)