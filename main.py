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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_params(kc.PARAMS_PATH)

### Stable baselines
def train_butterfly_supersuit(env, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in a cooperative Parallel environment

    env.reset(seed=seed)
    env.reward_table = []

    print(f"[SUCCESS] Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, 1, num_cpus=2, base_class="stable_baselines3")

    tuned_params = {
        "gamma": 0.9,
        "learning_rate": 1e-3,
    }

    model = PPO(
        "MlpPolicy",
        env,
        verbose = 1,
        n_steps = 10,
        batch_size=10,
        **tuned_params
    )

    """model.learn(total_timesteps=70000)"""

    """model = DQN(
        env=env,
        policy="MlpPolicy",
        #tensorboard_log="./board/",
        learning_rate=0.001,
        #tensorboard_log="./board/",
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0,
        verbose=1,
    )"""

    model.learn(total_timesteps=10)

    print(f"[SUCCESS] Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

def main():

    Sumo_sim=Sumo(params)
    Sumo_sim.Sumo_start()

    env = TrafficEnvironment(params[kc.ENVIRONMENT_PARAMETERS], params[kc.SIMULATION_PARAMETERS], params[kc.AGENTS_GENERATION_PARAMETERS])
    
    parallel_api_test(env, num_cycles=1_000_000)
    print("\n[SUCCESS] Passed parallel_api_test\n")

    env_kwargs = {}
    train_butterfly_supersuit(env, steps=100, seed=0, **env_kwargs) 

    Sumo_sim.Sumo_stop()  

main()