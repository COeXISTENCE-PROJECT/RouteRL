import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from .learning_model import BaseLearningModel
from .dqn import Network

class MAPPO(BaseLearningModel):
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        num_agents: int = 20,
        # --- policy settings ---
        shared_policy: bool = False,
        policy_nets: list[nn.Module] | None = None,
        policy_arch_kwargs: dict | None = None,
        # --- critic settings ---
        share_critic: bool = True,
        critic_nets: list[nn.Module] | None = None,
        critic_arch_kwargs: dict | None = None,
        # --- default architecture parameters ---
        default_widths: list[int] = [32, 64, 32],
        # --- hyperparameters ---
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.0003,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        batch_size: int = 64,
        memory_size: int = 5000,
        device: torch.device | None = None
    ):
        super().__init__()
        # device setup 
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.num_agents = num_agents

        # training phase flag
        self.training = True
        
        # hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.last_states = {}
        self.last_actions = {}
        self.last_log_probs = {}

        # architecture args
        ws = default_widths if policy_arch_kwargs is None else policy_arch_kwargs.get('widths', default_widths)

        # --- Policy networks ---
        if policy_nets is not None:
            assert len(policy_nets) == num_agents or shared_policy, \
                "If policy_nets is provided, it must match the number of agents or be shared."
            self.policies = [net.to(self.device) for net in (policy_nets if not shared_policy else [policy_nets[0]] * num_agents)]
        else:
            self.policies = []
            for _ in range(num_agents):
                net = Network(state_size, action_space_size, ws).to(self.device)
                self.policies.append(net)
            if shared_policy:
                self.policies = [self.policies[0]] * num_agents

        # create actor optimizers
        if shared_policy:
            self.actor_optimizer = optim.Adam(self.policies[0].parameters(), lr=lr_actor)
        else:
            self.actor_optimizer = [optim.Adam(policy.parameters(), lr=lr_actor) for policy in self.policies]
        
        # --- Critic networks ---
        if critic_nets is not None:
            assert len(critic_nets) == num_agents or share_critic, \
                "If critic_nets is provided, it must match the number of agents or be shared."
            self.critics = [net.to(self.device) for net in (critic_nets if not share_critic else [critic_nets[0]] * num_agents)]
        else:
            # build critics using generic Network class
            ch_ws = critic_arch_kwargs.get('widths', default_widths) if critic_arch_kwargs else default_widths
            self.critics = []
            for _ in range(num_agents):
                net = Network(state_size, 1, ch_ws).to(self.device)
                self.critics.append(net)
            if share_critic:
                shared_critic = self.critics[0]
                self.critics = [shared_critic] * num_agents

        # create critic optimizers
        if share_critic:
            self.critic_optim = optim.Adam(self.critics[0].parameters(), lr=lr_critic)
        else:
            self.critic_optim = [optim.Adam(net.parameters(), lr=lr_critic) for net in self.critics]

        # loss tracking
        self.loss_actor = []
        self.loss_critic = []

    def train(self):
        """Set all models to training mode"""
        for policy in self.policies:
            policy.train()
        for critic in self.critics:
            critic.train()
        self.training = True
        return self
    
    def eval(self):
        """Set all models to evaluation mode"""
        for policy in self.policies:
            policy.eval()
        for critic in self.critics:
            critic.eval()
        self.training = False
        return self

    def act(self, state: any, agent_id: int):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policies[agent_id](state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item() if self.training else torch.argmax(logits).item()
        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        self.last_states[agent_id] = state
        self.last_actions[agent_id] = action
        self.last_log_probs[agent_id] = log_prob
        return action

    def learn(
        self,
        states: list, actions: list, rewards: list,
        old_log_probs: list, next_states: list, dones: list, agent_ids: list
    ):
        for s, a, r, lp, ns, d, aid in zip(states, actions, rewards, old_log_probs, next_states, dones, agent_ids):
            self.memory.append((s, a, r, lp, ns, d, aid))

        if len(self.memory) < self.batch_size: return
        
        batch = random.sample(self.memory, self.batch_size)
        s_batch, a_batch, r_batch, lp_batch, ns_batch, d_batch, id_batch = zip(*batch)

        states_tensor = torch.FloatTensor(s_batch).to(self.device)
        actions_tensor = torch.LongTensor(a_batch).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(r_batch).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(ns_batch).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(lp_batch).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(d_batch).unsqueeze(1).to(self.device)

        id_tensor = torch.LongTensor(id_batch)
        unique_ids = id_tensor.unique()

        total_policy_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_count = 0

        for aid in unique_ids.tolist():
            mask = (id_tensor == aid)
            states_tensor_a = states_tensor[mask]
            actions_tensor_a = actions_tensor[mask]
            rewards_tensor_a = rewards_tensor[mask]
            next_states_tensor_a = next_states_tensor[mask]
            old_log_probs_tensor_a = old_log_probs_tensor[mask]
            dones_tensor_a = dones_tensor[mask]

            # critic forward
            values = self.critics[aid](states_tensor_a)
            next_values = self.critics[aid](next_states_tensor_a)
            with torch.no_grad():
                targets = rewards_tensor_a + self.gamma * next_values * (1 - dones_tensor_a)
            critic_loss = nn.MSELoss()(values, targets)

            # policy forward
            logits = self.policies[aid](states_tensor_a)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_tensor_a.squeeze(1)).unsqueeze(1)
            ratios = torch.exp(new_log_probs - old_log_probs_tensor_a)
            clipped_ratios = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -torch.min(ratios * (targets - values.detach()), clipped_ratios * (targets - values.detach())).mean()
            entropy = dist.entropy().mean()

            batch_size_a = mask.sum().item()
            total_critic_loss += critic_loss * batch_size_a
            total_policy_loss += policy_loss * batch_size_a
            total_entropy += entropy * batch_size_a
            total_count += batch_size_a

        # average losses
        avg_critic_loss = total_critic_loss / total_count
        avg_policy_loss = total_policy_loss / total_count
        avg_entropy = total_entropy / total_count

        # update critic
        if isinstance(self.critic_optim, list):
            for aid in unique_ids.tolist():
                self.critic_optim[aid].zero_grad()
            avg_critic_loss.backward()
            for aid in unique_ids.tolist():
                self.critic_optim[aid].step()
        else:
            self.critic_optim.zero_grad()
            avg_critic_loss.backward()
            self.critic_optim.step()
        self.loss_critic.append(avg_critic_loss.item())

        # update actor
        total_loss = avg_policy_loss - self.entropy_coef * avg_entropy
        if isinstance(self.actor_optimizer, list):
            for aid in unique_ids.tolist():
                self.actor_optimizer[aid].zero_grad()
            total_loss.backward()
            for aid in unique_ids.tolist():
                self.actor_optimizer[aid].step()
        else:
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
        self.loss_actor.append(avg_policy_loss.item())

    def get_last_observation(self, agent_id: int):
        return self.last_states.get(agent_id, None)
    
    def get_last_action(self, agent_id: int):
        return self.last_actions.get(agent_id, None)
    
    def get_last_log_prob(self, agent_id: int):
        return self.last_log_probs.get(agent_id, None)
    
    def get_policy(self, agent_id: int):
        return self.policies[agent_id] if agent_id < len(self.policies) else None