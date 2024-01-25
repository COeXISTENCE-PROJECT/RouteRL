from environment import TrafficEnvironment
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from typing import Callable, Optional, Tuple, Union

###
### Took it from sumo-rl ###
###

def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class SumoEnvironmentPZ(AECEnv, EzPickle):

    """A wrapper for the SUMO environment that implements the AECEnv interface from PettingZoo.

    For more information, see https://pettingzoo.farama.org/api/aec/.

    The arguments are the same as for :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "name": "sumo_rl_v0", "is_parallelizable": True}

    def __init__(self, **kwargs):
        """Initialize the environment."""
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = TrafficEnvironment(**self._kwargs)

        self.agents = self.env.ts_ids
        self.possible_agents = self.env.ts_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def seed(self, seed=None):
        """Set the seed for the environment."""
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.compute_info()

    def compute_info(self):
        """Compute the info for the current step."""
        self.infos = {a: {} for a in self.agents}
        infos = self.env._compute_info()
        for a in self.agents:
            for k, v in infos.items():
                if k.startswith(a) or k.startswith("system"):
                    self.infos[a][k] = v

    def observation_space(self, agent):
        """Return the observation space for the agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Return the action space for the agent."""
        return self.action_spaces[agent]

    def observe(self, agent):
        """Return the observation for the agent."""
        obs = self.env.observations[agent].copy()
        return obs

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        self.env.close()

    def render(self):
        """Render the environment."""
        return self.env.render()

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file."""
        self.env.save_csv(out_csv_name, episode)

    def step(self, action):
        """Step the environment."""
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception(
                "Action for agent {} must be in Discrete({})."
                "It is currently {}".format(agent, self.action_spaces[agent].n, action)
            )

        self.env._apply_actions({agent: action})

        if self._agent_selector.is_last():
            self.env._run_steps()
            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.compute_info()
        else:
            self._clear_rewards()

        done = self.env._compute_dones()["__all__"]
        self.truncations = {a: done for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()


###
### Took it from chat-gpt ###
###


class MultiAgentEnvWrapper(gym.Env):
    def __init__(self, your_multi_agent_environment):
        super(MultiAgentEnvWrapper, self).__init__()

        # Assuming each agent has its own action and observation space
        self.action_space = [0, 1, 2]
        # Any additional initialization code specific to your multi-agent environment

        self.multi_agent_env = your_multi_agent_environment

    def reset(self):
        return self.multi_agent_env.reset()

    def step(self, actions):
        # Assuming 'actions' is a dictionary where keys are agent IDs and values are their corresponding actions
        observations, rewards, dones, infos = self.multi_agent_env.step(actions)
        
        # You can modify the return values based on your specific multi-agent environment's structure
        return observations, rewards, dones, infos

    def render(self, mode='human'):
        return self.multi_agent_env.render(mode)

    def close(self):
        return self.multi_agent_env.close()
