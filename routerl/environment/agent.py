"""
This module contains the human and machine agent classes, which represent vehicles driving
from an origin to a destination in the simulation.
"""

import numpy as np

from abc import ABC, abstractmethod

from routerl.keychain import Keychain as kc
from routerl.human_learning import get_learning_model


class BaseAgent(ABC):
    """This is the abstract base class for the human and machine agent classes.

    Args:
        id (int): 
            The id of the agent.
        kind (str): 
            The kind of the agent (Human or AV).
        start_time (float): 
            The start time of the agent.
        origin (float): 
            The origin of the agent.
        destination (float): 
            The destination value of the agent.
        behavior (float): 
            The behavior of the agent.
    """

    def __init__(self, id, kind, start_time, origin, destination, behavior):
        self.id = id
        self.kind = kind
        self.start_time = start_time
        self.origin = origin
        self.destination = destination
        self.behavior = behavior
        self.last_action = 0
        self.default_action = None

    @property
    @abstractmethod
    def last_reward(self) -> None:
        """Return the last reward of the agent.

        Returns:
            None
        """

        pass
    
    @last_reward.setter
    @abstractmethod
    def last_reward(self, reward) -> None:
        """Set the last reward of the agent.

        Args:
            reward (float): The reward of the agent.
        Returns:
            None
        """

        pass

    @abstractmethod
    def act(self, observation) -> None:
        """Pick action according to your knowledge, or randomly.

        Args:
            observation (float): The observation of the agent.
        Returns:
            None
        """

        pass

    @abstractmethod
    def learn(self, action, observation) -> None:
        """Pass the applied action and reward once the episode ends, and it will remember the consequences.

        Args:
            action (int): The action taken by the agent.
            observation (float): The observation of the agent.
        Returns:
            None
        """

        pass

    @abstractmethod
    def get_state(self, observation) -> None:
        """Return the state of the agent, given the observation

        Args:
            observation (float): The observation of the agent.
        Returns:
            None
        """

        pass

    @abstractmethod
    def get_reward(self, observation) -> None:
        """Derive the reward of the agent, given the observation

        Args:
            observation (float): The observation of the agent.
        Returns:
            None
        """

        pass


class HumanAgent(BaseAgent):
    """Class representing human drivers, responsible for modeling their learning process
    and decision-making in route selection.

    Args:
        id (int):
            The id of the agent.
        start_time (float):
            The start time of the agent.
        origin (float):
            The origin of the agent.
        destination (float):
            The destination value of the agent.
        params (dict):
            The parameters for the learning model of the agent as specified in `here <https://coexistence-project.github.io/RouteRL/documentation/pz_env.html#>`_.
        initial_knowledge (float):
            The initial knowledge of the agent.
    """

    def __init__(self, id, start_time, origin, destination, params, initial_knowledge):
        kind = kc.TYPE_HUMAN
        behavior = kc.SELFISH
        super().__init__(id, kind, start_time, origin, destination, behavior)
        self.initial_knowledge = initial_knowledge
        self.model = get_learning_model(params, initial_knowledge)
        self.last_reward = None

    def __repr__(self):
        return f"Human {self.id}"

    @property
    def last_reward(self) -> float:
        """Set the last reward of the agent.

        Returns:
            float: The last reward of the agent.
        """

        return self._last_reward

    @last_reward.setter
    def last_reward(self, reward) -> None:
        """Set the last reward of the agent.

        Args:
            reward (float): The reward of the agent.
        Returns:
            None
        """

        self._last_reward = reward

    def act(self, observation) -> int:  
        """Returns the agent's action (route of choice) based on the current observation from the environment.

        Args:
            observation (list): The observation of the agent.
        Returns:
            int: The action of the agent.
        """
        if self.default_action is not None:
            return self.default_action
        return self.model.act(observation)

    def learn(self, action, observation) -> None:
        """Updates the agent's knowledge based on the action taken and the resulting observations.

        Args:
            action (int): The action of the agent.
            observation (list[dict]): The observation of the agent.
        Returns:
            None
        """

        reward = self.get_reward(observation)
        self.last_reward = reward
        self.model.learn(None, action, reward)

    def get_state(self, _) -> None:
        """Returns the current state of the agent.

        Args:
            _ (Any): The current state of the agent.
        Returns:
            None
        """

        return None

    def get_reward(self, observation: list[dict]) -> float:
        """This function calculated the reward of each individual agent.

        Args:
            observation (list[dict]): The observation of the agent.
        Returns:
            float: Own travel time of the agent.
        """

        own_tt = -1 * next(obs[kc.TRAVEL_TIME] for obs in observation if obs[kc.AGENT_ID] == self.id)
        return own_tt
    

class MachineAgent(BaseAgent):
    """A class that models Autonomous Vehicles (AVs), focusing on their learning mechanisms
    and decision-making processes for selecting optimal routes.

    Args:
        id (int): 
            The id of the agent.
        start_time (float): 
            The start time of the agent.
        origin (float): 
            The origin of the agent.
        destination (float): 
            The destination value of the agent.
        params (dict): 
            The parameters of the machine agent as specified in `here <https://coexistence-project.github.io/RouteRL/documentation/pz_env.html#>`_.
        action_space_size (int): 
            The size of the action space of the agent.
    """

    def __init__(self, id, start_time, origin, destination, params, action_space_size):
        kind = kc.TYPE_MACHINE
        behavior = params[kc.BEHAVIOR]
        super().__init__(id, kind, start_time, origin, destination, behavior)
        self.observed_span = params[kc.OBSERVED_SPAN]
        self.action_space_size = action_space_size
        self.state_size = action_space_size * 2
        self.model = None
        self.last_reward = None
        self.rewards_coefs = self._get_reward_coefs()

    def __repr__(self) -> str:
        machine_id = f"Machine {self.id}"

        return machine_id

    @property
    def last_reward(self) -> float:
        """Set the last reward of the agent.

        Returns:
            float: The last reward of the agent.
        """

        return self._last_reward

    @last_reward.setter
    def last_reward(self, reward) -> None:
        """Sets the last reward of the agent.

        Args:
            reward (float): The reward of the agent.
        Returns:
            None
        """

        self._last_reward = reward

    def act(self, _) -> None:
        """**Deprecated**

        Args:
            _ (Any): The current state of the agent.
        Returns:
            None
        """

        return None

    def learn(self, _) -> None:
        """**Deprecated**
        
        Args:
            _ (Any): The current state of the agent.
            
        Returns:
            None
        """

        return None

    def get_state(self, observation: list[dict]) -> list[int]:
        """Generates the current state representation based on recent observations of agents navigating
        from the same origin to the same destination.

        Args:
            observation (list[dict]): The recent observations of the agent.
            
        Returns:
            list[int]: The current state representation.
        """

        min_start_time = self.start_time - self.observed_span
        human_prior, machine_prior = list(), list()
        for obs in observation:
            if ((obs[kc.AGENT_ORIGIN], obs[kc.AGENT_DESTINATION]) == (self.origin, self.destination)
                    and (obs[kc.AGENT_START_TIME] > min_start_time)
            ):
                if obs[kc.AGENT_KIND] == kc.TYPE_HUMAN:
                    human_prior.append(obs)
                elif obs[kc.AGENT_KIND] == kc.TYPE_MACHINE:
                    machine_prior.append(obs)

        warmth_human = [0] * (self.state_size // 2)
        warmth_machine = [0] * (self.state_size // 2)
        
        if human_prior:
            for row in human_prior:
                action = row[kc.ACTION]
                start_time = row[kc.AGENT_START_TIME]
                warmth = start_time - min_start_time
                warmth_human[action] += warmth
                
        if machine_prior:
            for row in machine_prior:
                action = row[kc.ACTION]
                start_time = row[kc.AGENT_START_TIME]
                warmth = start_time - min_start_time
                warmth_machine[action] += warmth

        warmth_agents = warmth_human + warmth_machine
        return warmth_agents

    def get_reward(self, observation: list[dict], group_vicinity: bool = False) -> float:
        """This method calculated the reward of each individual agent, based on the travel time of the agent,
        the group of agents, the other agents, and all agents, weighted according to the agent's behavior.

        Args:
            observation (list[dict]): The current observation of the agent.
                
        Returns:
            float: The reward of the agent.
        """
        vicinity_obs = list()
        if group_vicinity == True:

            min_start_time, max_start_time = self.start_time - self.observed_span, self.start_time + self.observed_span

            for obs in observation:
                if (obs[kc.AGENT_ORIGIN], obs[kc.AGENT_DESTINATION]) == (self.origin, self.destination):
                    if min_start_time <= obs[kc.AGENT_START_TIME] <= max_start_time:
                        vicinity_obs.append(obs)
        else:        
            for obs in observation:
                vicinity_obs.append(obs)

        group_obs, others_obs, all_obs, own_tt = list(), list(), list(), None
        for obs in vicinity_obs:
            all_obs.append(obs[kc.TRAVEL_TIME])
            if obs[kc.AGENT_KIND] == self.kind:
                group_obs.append(obs[kc.TRAVEL_TIME])
            else:
                others_obs.append(obs[kc.TRAVEL_TIME])
            if obs[kc.AGENT_ID] == self.id:
                own_tt = obs[kc.TRAVEL_TIME]
                
        group_tt = np.mean(group_obs) if group_obs else 0
        others_tt = np.mean(others_obs) if others_obs else 0
        all_tt = np.mean(all_obs) if all_obs else 0
        
        a, b, c, d = self.rewards_coefs
        agent_reward  = a * own_tt + b * group_tt + c * others_tt + d * all_tt
        return agent_reward

    def _get_reward_coefs(self) -> tuple:
        a, b, c, d = 0, 0, 0, 0
        if self.behavior == kc.SELFISH:
            a, b, c, d = -1, 0, 0, 0 
        elif self.behavior == kc.COMPETITIVE:
            a, b, c, d = -2, 0, 1, 0
        elif self.behavior == kc.COLLABORATIVE:
            a, b, c, d = -0.5, -0.5, 0, 0
        elif self.behavior == kc.COOPERATIVE:
            a, b, c, d = 0, -1, 0, 0
        elif self.behavior == kc.SOCIAL:
            a, b, c, d = -0.5, 0, 0, -0.5
        elif self.behavior == kc.ALTRUISTIC:
            a, b, c, d = 0, 0, 0, -1
        elif self.behavior == kc.MALICIOUS:
            a, b, c, d = 0, 0, 1, 0
        return a, b, c, d