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


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_params(kc.PARAMS_PATH)

### Stable baselines
def train_butterfly_supersuit(env, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in a cooperative Parallel environment

    env.reset(seed=seed)

    print(f"[SUCCESS] Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, 1, num_cpus=2, base_class="stable_baselines3")

    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[64, 64, 64])

    model = PPO(
        "MlpPolicy",
        env,
        verbose = 1,
        n_steps = 10,
        batch_size=10,
        device = "cuda",
        policy_kwargs=policy_kwargs,
        gamma = 0.9,
        learning_rate = 1e-3,
        tensorboard_log="./board/",
    )

    model.learn(total_timesteps=2000)

    """model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0,
        verbose=1,
        device="cuda",
        tensorboard_log="./board/",
    )

    model.learn(total_timesteps=200000)"""

    print(f"[SUCCESS] Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

    # Enjoy trained agent
    obs = env.reset()
    for i in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)


def main():

    Sumo_sim=Sumo(params)
    Sumo_sim.Sumo_start()

    env = TrafficEnvironment(params[kc.ENVIRONMENT_PARAMETERS], params[kc.SIMULATION_PARAMETERS], params[kc.AGENTS_GENERATION_PARAMETERS])
    print("[SUCCESS] Environment initiated!")
    
    #parallel_api_test(env, num_cycles=1_000_000)
    #print("\n[SUCCESS] Passed parallel_api_test\n")

    #env_kwargs = {}
    #train_butterfly_supersuit(env, steps=100, seed=0, **env_kwargs) 

    env = PettingZooWrapper(
        env=env,
        return_state=True,
        group_map=None, # Use default for parallel (all pistons grouped together)
    )

    print(env.group_map)

    env.rollout(10)
    num_cells = 256
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    print(env.action_spec)

    check_env_specs(env)

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
        actor_net, in_keys=["agents", "observation"], out_keys=["loc", "scale"]
    )

    #print("env.action_spec.space is: ", env.action_spec.space)

    """policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.low,
            "max": env.action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )"""
        

    Sumo_sim.Sumo_stop()  

main()