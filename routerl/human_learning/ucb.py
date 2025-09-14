import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from .learning_model import BaseLearningModel

class UCB(BaseLearningModel):
    """A simple tabular UCB agent."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_agents: int,
        alpha: float,
        beta: float,
        seed: int | None = None,
    ):
        """Initializes the UCB agent with the given parameters.

        Creates the necessary data structures for counting visits and transitions,
        as well as storing estimates for rewards and the value function.

        Args:
            num_states (int): Number of states in the environment.
            num_actions (int): Number of possible actions.
            alpha (float): Step size to learn the value function.
            beta (float): Exploration bonus coefficient.
            seed (int | None, optional): Seed to ensure reproducibility. Defaults to None.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.obs_dim = tuple([num_agents] * num_actions)

        self.sa_counts = np.zeros((num_states, num_actions), dtype=np.int32)

        self.alpha = alpha
        self.beta = beta

        self.global_step = 1 # starting at 1 to avoid ln(0)

        self.Q = np.full(
            (num_states, num_actions),
            dtype=np.float32,
            fill_value=0,
        )
        np.random.seed(seed)
    
    def learn(self, state, action, reward):
        """Updates the internal model with a new experience.

        Increments visit counts for the given state-action pair, adds
        the observed reward, and updates transition counts.

        Args:
            action (int): Action taken in the current state.
            obs (int): Current state.
            reward (float): Reward received upon transitioning to next_state.
        """
        obs_idx = np.ravel_multi_index(state, self.obs_dim)

        self.sa_counts[obs_idx, action] += 1
        self.global_step += 1

        old_estimate = self.Q[obs_idx, action]
        self.Q[obs_idx, action] = old_estimate + self.alpha * (reward - old_estimate)

    def act(self, obs) -> int:
        """Selects an action based on the current value function.

        Computes the estimated Q-value for each possible action and
        returns the action that yields the highest value.

        Args:
            obs (int): Current state from which to choose an action.

        Returns:
            int: The action that maximizes the estimated Q-value.
        """
        obs_idx = np.ravel_multi_index(obs, self.obs_dim)
        self.last_obs = obs_idx
        division_coef = 0.001

        values = self.Q[obs_idx] + self.beta * np.sqrt(np.log(self.global_step) / (self.sa_counts[obs_idx] + division_coef))
        return np.random.choice(
            np.argwhere(values == np.max(values)).reshape((-1,))
        )