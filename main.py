from environment import TrafficEnvironment
from MultiAgentWrapper import MultiAgentEnvWrapper

#import gymnasium as gym
from keychain import Keychain as kc
from services import Trainer
from services import create_agent_objects
from services import confirm_env_variable
from services import get_json
#from stable_baselines3 import DQN
#from stable_baselines3.common.vec_env import DummyVecEnv



confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_json(kc.PARAMS_PATH)

"""
Next Improvement:
1. First, determine number of agents first (read from params.json)
2. Pass this to simulator
3. Calculate the freeflow travel times
4. Only then generate agents, by also using freeflow information for initial knowledge for human agents
"""

def main():

    env = TrafficEnvironment(params[kc.SIMULATION_PARAMETERS]) # pass some params for the simulation
    agents = create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS], env.calculate_free_flow_times())

    # Wrap your multi-agent environment with the Gym wrapper
    """gym_multi_agent_env = MultiAgentEnvWrapper(env)

    gym_multi_agent_env = DummyVecEnv([lambda: MultiAgentEnvWrapper(env)])



    model = DQN('MlpPolicy', env, verbose=1)"""

    trainer = Trainer(params[kc.TRAINING_PARAMETERS])
    agents = trainer.train(env, agents)
    env.plot_rewards()


main()