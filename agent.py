import numpy as np
import pandas as pd
import random

from abc import ABC, abstractmethod

from keychain import Keychain as kc
from utilities import list_to_string


class Agent(ABC):

    """
    This is an abstract class for agents, to be inherited by specific type of agent classes
    It is not to be instantiated, but to provide a blueprint for all types of agents
    """
    
    def __init__(self, id, start_time, origin, destination):
        self.id = id

        self.start_time = start_time
        self.origin = origin
        self.destination = destination

    @abstractmethod
    def act(self, observation):  
        # Pick action according to your knowledge, or randomly
        pass

    @abstractmethod
    def get_state(self, observation):
        # Return the state of the agent, given the observation
        pass

    @abstractmethod
    def learn(self, action, observation):
        # Pass the applied action and reward once the episode ends, and it will remember the consequences
        pass

    @abstractmethod
    def get_reward(self, observation):
        # Return the reward of the agent, given the observation
        pass




class HumanAgent(Agent):

    def __init__(self, id, start_time, origin, destination, params, initial_knowledge, mutate_to=None, mutate_type=None):
        super().__init__(id, start_time, origin, destination)

        self.kind = kc.TYPE_HUMAN
        self.mutate_to = mutate_to
        self.mutate_type = mutate_type

        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = random.uniform(params[kc.BETA] - beta_randomness, params[kc.BETA] + beta_randomness)
        self.alpha = params[kc.ALPHA]

        self.cost = np.array(initial_knowledge, dtype=float)


    def act(self, observation):  
        #state = self.get_state(observation)
        utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        prob_dist = [self.calculate_prob(utilities, j) for j in range(len(self.cost))]
        action = np.random.choice(list(range(len(self.cost))), p=prob_dist) 
        return action


    def get_state(self, observation):
        return list_to_string(observation, separator="_")        


    def learn(self, action, observation):
        reward = self.get_reward(observation)
        self.cost[action] = ((1-self.alpha) * self.cost[action]) + (self.alpha * reward)


    def get_reward(self, observation):
        reward = observation.loc[observation[kc.AGENT_ID] == self.id, kc.TRAVEL_TIME].item()
        return reward


    def calculate_prob(self, utilities, n):
        prob = utilities[n] / sum(utilities)
        return prob
    

    def mutate(self):
        self.mutate_to.receive_initial_knowledge(self.cost)
        return self.mutate_to
    



class MachineAgent(Agent):

    def __init__(self, id, start_time, origin, destination, params, action_space_size):
        super().__init__(id, start_time, origin, destination)

        self.kind = kc.TYPE_MACHINE

        self.epsilon = params[kc.EPSILON]
        self.epsilon_decay_rate = params[kc.EPSILON_DECAY_RATE]
        self.alpha = params[kc.ALPHA]
        self.gamma = params[kc.GAMMA]

        self.action_space_size = action_space_size

        self.sample_q_table_row = np.zeros((action_space_size))
        self.q_table = pd.DataFrame(columns=[kc.STATE, kc.Q_TABLE])

        self.last_state = None


    def act(self, observation):
        state = self.get_state(observation)
        self.ensure_row_in_q_table(state)
        self.last_state = state
        if np.random.rand() < self.epsilon:    # Explore
            return np.random.choice(self.action_space_size)
        else:    # Exploit
            table = self.q_table.loc[self.q_table[kc.STATE] == state, kc.Q_TABLE].item()
            return np.argmin(table)
        

    def get_state(self, observation):
        return list_to_string(observation, separator="_")


    def ensure_row_in_q_table(self, state):
        if not (state in self.q_table[kc.STATE].values):
            self.q_table.loc[len(self.q_table.index)] = {kc.STATE: state, kc.Q_TABLE: self.sample_q_table_row}
                

    def learn(self, action, observation):
        reward = self.get_reward(observation)
        table = self.q_table.loc[self.q_table[kc.STATE] == self.last_state, kc.Q_TABLE].item()
        prev_knowledge = table[action]
        table[action] = prev_knowledge + (self.alpha * (reward - prev_knowledge))
        self.decay_epsilon()
        

    def get_reward(self, observation):
        reward = observation.loc[observation[kc.AGENT_ID] == self.id, kc.TRAVEL_TIME].item()
        return reward


    def decay_epsilon(self):    # Slowly become deterministic
        self.epsilon *= self.epsilon_decay_rate

    
    def receive_initial_knowledge(self, initial_knowledge):
        self.sample_q_table_row = initial_knowledge




class MaliciousMachineAgent(Agent):

    def __init__(self, id, start_time, origin, destination, params, action_space_size):
        super().__init__(id, start_time, origin, destination)

        self.kind = kc.TYPE_MACHINE_2

        self.epsilon = params[kc.EPSILON]
        self.epsilon_decay_rate = params[kc.EPSILON_DECAY_RATE]
        self.alpha = params[kc.ALPHA]
        self.gamma = params[kc.GAMMA]

        self.action_space_size = action_space_size
        
        self.sample_q_table_row = np.zeros((action_space_size))
        self.q_table = pd.DataFrame(columns=[kc.STATE, kc.Q_TABLE])

        self.last_state = None


    def act(self, observation):
        state = self.get_state(observation)
        self.ensure_row_in_q_table(state)
        self.last_state = state
        if np.random.rand() < self.epsilon:    # Explore
            return np.random.choice(self.action_space_size)
        else:    # Exploit
            table = self.q_table.loc[self.q_table[kc.STATE] == state, kc.Q_TABLE].item()
            return np.argmax(table)     # Malicious machines always choose the max
        

    def get_state(self, observation):
        state = [0] * self.action_space_size

        if not observation.empty:
            for idx, row in observation.iterrows():
                action = row[kc.ACTION]
                start_time = row[kc.AGENT_START_TIME]
                state[action] += start_time
            if sum(state):
                state = [x / sum(state) for x in state]
            state = [round(p / 0.05) * 0.05 for p in state]

        return list_to_string(state, separator="_")


    def ensure_row_in_q_table(self, state):
        if not (state in self.q_table[kc.STATE].values):
            self.q_table.loc[len(self.q_table.index)] = {kc.STATE: state, kc.Q_TABLE: self.sample_q_table_row}
                

    def learn(self, action, observation):
        reward = self.get_reward(observation)
        table = self.q_table.loc[self.q_table[kc.STATE] == self.last_state, kc.Q_TABLE].item()
        prev_knowledge = table[action]
        table[action] = prev_knowledge + (self.alpha * (reward - prev_knowledge))
        self.decay_epsilon()
        

    def get_reward(self, observation):
        reward = observation.loc[observation[kc.AGENT_ID] == self.id, kc.TRAVEL_TIME].item()
        return reward


    def decay_epsilon(self):    # Slowly become deterministic
        self.epsilon *= self.epsilon_decay_rate

    
    def receive_initial_knowledge(self, initial_knowledge):
        self.sample_q_table_row = initial_knowledge




import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_space_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepMachineAgent(Agent):
    def __init__(self, id, start_time, origin, destination, params, action_space_size):
        super().__init__(id, start_time, origin, destination)

        self.kind = kc.TYPE_MACHINE

        self.epsilon = params[kc.EPSILON]
        self.epsilon_decay_rate = params[kc.EPSILON_DECAY_RATE]
        self.action_space_size = action_space_size
        self.state_size = action_space_size

        self.memory = deque(maxlen=256)
        self.batch_size = 32
        self.target_update_frequency = 3

        self.q_network = DQN(self.state_size, action_space_size)
        self.target_network = DQN(self.state_size, action_space_size)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.03)
        self.loss_fn = nn.MSELoss()

        self.last_state = None
        self.last_action = None

    def act(self, observation):
        state = self.get_state(observation)
        self.last_state = state
        if np.random.rand() < self.epsilon:    # Explore
            self.last_action = np.random.choice(self.action_space_size)
        else:    # Exploit
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            self.last_action = torch.argmax(q_values).item()
        return self.last_action

    def get_state(self, observation):
        #return list_to_string(observation, separator="_")
        return observation

    def learn(self, action, observation):
        reward = self.get_reward(observation)
        next_state = [self.get_state(observation)]
        done = True  # This should be set to True if the episode is finished

        self.memory.append((self.last_state, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states_tensor).max(1)[0].unsqueeze(1)
            target_q_values = rewards_tensor + self.gamma * max_next_q_values * (1 - dones_tensor)

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.decay_epsilon()
        if self.target_update_frequency == 0:
            self.target_update_frequency = 3
            self.update_target_network()
        else:
            self.target_update_frequency -= 1

    def get_reward(self, observation):
        reward = observation.loc[observation[kc.AGENT_ID] == self.id, kc.TRAVEL_TIME].item()
        return reward

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def receive_initial_knowledge(self, initial_knowledge):
        pass
