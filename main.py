from environment import TrafficEnvironment
from keychain import Keychain as kc
from utilities import confirm_env_variable
from utilities import get_params
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from pettingzoo.test import parallel_api_test
from stable_baselines3 import PPO
import supersuit as ss
from Sumo_controller import Sumo
import os
import torch as th
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import TransformedEnv, RewardSum
from torchrl.envs.utils import check_env_specs
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from tensordict.nn.distributions import NormalParamExtractor
import torch
from torch import nn
from tensordict.nn import TensorDictModule
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_params(kc.PARAMS_PATH)




def main():

    Sumo_sim=Sumo(params)
    Sumo_sim.Sumo_start()

    env = TrafficEnvironment(params[kc.ENVIRONMENT_PARAMETERS], params[kc.SIMULATION_PARAMETERS], params[kc.AGENTS_GENERATION_PARAMETERS])
    print("[SUCCESS] Environment initiated!")

    ## https://github.com/pytorch/rl/blob/main/torchrl/envs/libs/pettingzoo.py
    env = PettingZooWrapper(
        env=env,
        return_state=True,
        use_mask=True,
        group_map=None, # Use default for parallel
        categorical_actions=True,
    )
    print("env is: ", env)

    # This will modify the inputs and outputs of our environment in some way.
    # This specific one will sum rewards over the episodes.
    # We will tell the transform where where to find the reward key and where to write the summer episode reward
    # The transformed environment will inherit the device and meta-data of the wrapped environment and transform these depending on the sequence of transforms it contains.
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    print("Environment group mapping", env.group_map)

    rollout = env.rollout(1)
    print("rollout of one step:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)

    num_cells = 256
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    #print("Environement action space is: ", env.action_spec)

    check_env_specs(env)
    print("Observation space is: ", env.observation_spec)

    ### Policy

    share_parameters_policy = True

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["1", "observation"].shape[-1],  # n_obs_per_agent
            n_agent_outputs=2 * env.action_spec.shape[-1],  # 2 * n_actions_per_agents
            n_agents=env.n_agents,
            centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
            share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
    )

    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * 1, device=device),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=["1", "observation"], out_keys=["loc", "scale"]
    )

    #print("env.action_spec.space is: ", env.action_spec.space)

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=[("1", "loc"), ("1", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        return_log_prob=True,
        log_prob_key=("1", "sample_log_prob"),
    )

    share_parameters_critic = True
    mappo = True  # IPPO if False

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["1", "observation"].shape[-1],
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.n_agents,
        centralised=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("1", "observation")],
        out_keys=[("1", "state_value")],
    )

    print("Running policy:", policy(env.reset()))
    #print("Running value:", critic(env.reset()))
            

    Sumo_sim.Sumo_stop()  

main()