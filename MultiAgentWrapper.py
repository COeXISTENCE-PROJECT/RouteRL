import gymnasium as gym
from gymnasium import spaces

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
