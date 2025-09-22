import random
import numpy as np
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from routerl import MAPPO
from routerl import TrafficEnvironment
from routerl import Keychain as kc
from hyperparameters_tuning import run_gridsearch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ENV_SEEDS = [0, 42, 2137]
TORCH_SEEDS = [7, 420, 1248]
TRAINING_EPISODES = 2000
HUMAN_LEARNING_EPISODES = 1

ALL_PARAM_GRIDS = {
    "lr_actor": [1e-1, 1e-10],
    "lr_critic": [1e-1, 1e-4, 1e-10],
    "clip_ratio": [0.1, 0.2, 0.3],
    "entropy_coef": [0.01, 0.05, 0.1],
    "value_coef": [0.1, 0.5, 1.0],
    "batch_size": [16, 64],
    "memory_size": [500, 2000],
    "shared_policy": [True, False],
    "share_critic": [True, False],
}

selected = [
    "memory_size",
    "share_critic",
    "batch_size",
]

env_params = {
    "agent_parameters": {
        "new_machines_after_mutation": 10,
        "human_parameters": {
            "model": "general_model",
            "noise_weight_agent": 0,
            "noise_weight_path": 0.8,
            "noise_weight_day": 0.2,
            "beta": -1,
            "beta_k_i_variability": 0.1,
            "epsilon_i_variability": 0.1,
            "epsilon_k_i_variability": 0.1,
            "epsilon_k_i_t_variability": 0.1,
            "greedy": 0.9,
            "gamma_c": 0.0,
            "gamma_u": 0.0,
            "remember": 1,
            "alpha_zero": 0.8,
            "alphas": [0.2]
        },
        "machine_parameters": {
            "behavior": "selfish",
            "observation_type": "previous_agents_plus_start_time",
        }
    },
    "simulator_parameters": {
        "network_name": "two_route_yield",
        "sumo_type": "sumo",
    },
    "plotter_parameters": {
        "phases": [0, HUMAN_LEARNING_EPISODES, int(TRAINING_EPISODES) + HUMAN_LEARNING_EPISODES],
        "smooth_by": 50,
        "phase_names": ["Human learning", "Mutation - Machine learning", "Testing phase"],
        "plot_choices": "basic"
    },
    "path_generation_parameters": {
        "number_of_paths": 2,
        "beta": -1,
        "visualize_paths": True
    }
}

def init_rouerl_env(env_seed: int,
                     torch_seed: int,
                     env_params: dict,
                     mappo_kwargs: dict,
                     human_learning_episodes: int = 1) -> TrafficEnvironment:
    """
    1) Seeds SUMO/PettingZoo + PyTorch,
    2) Initialiazes the environment with the given parameters,
    3) Runs the human learning phase,
    4) Applies mutation and wraps mutated vehicles with MAPPO.
    
    Returns:
        env: TrafficEnvironment
        mappo: MAPPO instance
        id_to_idx: dict mapping raw agent IDs to their indices in the MAPPO model
    """

    # 1) Set seeds
    random.seed(env_seed)
    np.random.seed(env_seed)
    torch.manual_seed(torch_seed)

    # 2) Initialize the environment
    env = TrafficEnvironment(
        seed=env_seed,
        create_agents=False,
        create_paths=False,
        marginal_cost_calculation=False,
        **env_params
    )
    env.start()
    env.reset()

    # 3) Run the human learning phase
    for _ in range(human_learning_episodes):
        env.step()

    # 4) Apply mutation to the environment
    pre_mutation_agents = env.all_agents.copy()
    env.mutation_odd_id_agents()

    for h in env.human_agents:
        h.default_action = 0

    # 4) Initialize MAPPO
    num_agents = len(env.machine_agents)
    mappo = MAPPO(
        state_size=3,
        action_space_size=2,
        num_agents=num_agents,
        policy_arch_kwargs={'num_hidden': 1, 'widths': [32, 32]},
        critic_arch_kwargs={'num_hidden': 1, 'widths': [64, 64]},
        **mappo_kwargs
    )

    machines = env.machine_agents.copy()
    mutated_humans = dict()
    for machine in machines:
        for human in pre_mutation_agents:
            if human.id == machine.id:
                mutated_humans[str(machine.id)] = human
                break

    raw_ids = sorted(int(k) for k in mutated_humans.keys())
    id_to_idx = { raw_id: idx for idx, raw_id in enumerate(raw_ids) }

    return env, mappo, id_to_idx



def train_mappo(env_seed: int,
                torch_seed: int,
                env_params: dict,
                mappo_kwargs: dict) -> float:

    env, mappo, id_to_idx = init_rouerl_env(env_seed, torch_seed, env_params, mappo_kwargs)

    # Training
    for _ in range(TRAINING_EPISODES):
        env.reset()

        states, actions, rewards, logps, next_states, dones, agent_idxs = [], [], [], [], [], [], []

        for agent in env.agent_iter():
            raw_id = int(agent)
            idx = id_to_idx[raw_id]

            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                last_obs = mappo.get_last_observation(idx)
                last_action = mappo.get_last_action(idx)
                last_logp = mappo.get_last_log_prob(idx)

                states.append(last_obs)
                actions.append(last_action)
                rewards.append(reward)
                logps.append(last_logp)
                next_states.append([0, 0, 0])
                agent_idxs.append(idx)
                dones.append(1)
                action = None
            else:
                action = mappo.act(observation, idx)

            env.step(action)
                
            mappo.learn(states, actions, rewards, logps, next_states, dones, agent_idxs)

    # Testing
    total_reward = 0.0
    env.reset()

    mappo.eval()

    for agent in env.agent_iter():
        raw_id = int(agent)
        idx = id_to_idx[raw_id]

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
            # The reward is minus the travel time in this case.
            total_reward -= reward
        else:
            action = mappo.act(observation, idx)
        env.step(action)
    travel_time = total_reward / len(env.machine_agents)

    # Clean up
    env.stop_simulation()
    return travel_time

def main():
    run_gridsearch(
        train_fn=train_mappo,
        ALL_PARAM_GRIDS=ALL_PARAM_GRIDS,
        selected=selected,
        env_params=env_params,
        ENV_SEEDS=ENV_SEEDS,
        TORCH_SEEDS=TORCH_SEEDS,
        TRAINING_EPISODES=TRAINING_EPISODES, 
        HUMAN_LEARNING_EPISODES=HUMAN_LEARNING_EPISODES,
        project_name="MAPPO_Hyperparameter_Tuning"
    )

if __name__ == "__main__":
    main()