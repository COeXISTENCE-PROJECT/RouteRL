from environment import TrafficEnvironment
from MultiAgentWrapper import MultiAgentEnvWrapper

#import gymnasium as gym
from keychain import Keychain as kc
import os
from services import Trainer
from services import create_agent_objects
from services import confirm_env_variable
from services import get_json
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env




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
    ### agents - dict 
    # env.agents
    agents = create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS], env.calculate_free_flow_times())

    env.create_agents(agents)

    # Wrap your multi-agent environment with the Gym wrapper
    """gym_multi_agent_env = MultiAgentEnvWrapper(env)

    gym_multi_agent_env = DummyVecEnv([lambda: MultiAgentEnvWrapper(env)])"""


    #model = DQN('MlpPolicy', env, verbose=1)
    #model.learn(total_timesteps=25000)

    """register_env(
        env,
        lambda _: ParallelPettingZooEnv(
            (TrafficEnvironment(params[kc.SIMULATION_PARAMETERS])
            )
        ),
    )

    config = (
        PPOConfig()
        .environment(env, disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.95,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 100000},
        checkpoint_freq=10,
        local_dir="~/ray_results/",
        config=config.to_dict(),
    )"""


    ## env.trainer
    trainer = Trainer(params[kc.TRAINING_PARAMETERS])
    agents = trainer.train(env, agents)
    env.plot_rewards()


main()