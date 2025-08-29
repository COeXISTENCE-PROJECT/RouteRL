import random
import numpy as np
import torch
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import plot_optimization_history, plot_param_importances
import wandb
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from routerl import DQN
from routerl import TrafficEnvironment
from routerl import Keychain as kc


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

ENV_SEEDS = [0, 42, 2137]
TORCH_SEEDS = [7, 420, 1248]
TRAINING_EPISODES = 20 #2000
HUMAN_LEARNING_EPISODES = 1
TESTING_EPISODES = 10 #100

ALL_PARAM_GRIDS = {
    "learning_rate": [1e-1, 1e-4, 1e-10],
    "epsilon_decay_rate": [0.8, 0.99],
    "epsilon_min": [0.01, 0.05, 0.1],
    "epsilon_start": [0.9, 0.95, 0.99],
    "batch_size": [16, 32, 64],
    "memory_size": [500, 1000, 2000],
}

selected = [
    "epsilon_decay_rate",
    "batch_size",
    "memory_size"
]

new_machines_after_mutation = 10

env_params = {
    "agent_parameters" : {
        "new_machines_after_mutation": new_machines_after_mutation,

        "human_parameters" :
        {
            "model" : "general_model",

            "noise_weight_agent" : 0,
            "noise_weight_path" : 0.8,
            "noise_weight_day" : 0.2,

            "beta" : -1,
            "beta_k_i_variability" : 0.1,
            "epsilon_i_variability" : 0.1,
            "epsilon_k_i_variability" : 0.1,
            "epsilon_k_i_t_variability" : 0.1,

            "greedy" : 0.9,
            "gamma_c" : 0.0,
            "gamma_u" : 0.0,
            "remember" : 1,

            "alpha_zero" : 0.8,
            "alphas" : [0.2]  
        },
        "machine_parameters" :
        {
            "behavior" : "selfish",
            "observation_type" : "previous_agents_plus_start_time",
        }
    },
    "simulator_parameters" : {
        "network_name" : "two_route_yield",
        "sumo_type" : "sumo",
    },  
    "plotter_parameters" : {
        "phases" : [0, HUMAN_LEARNING_EPISODES, int(TRAINING_EPISODES) + HUMAN_LEARNING_EPISODES],
        "smooth_by" : 50,
        "phase_names" : [
            "Human learning", 
            "Mutation - Machine learning",
            "Testing phase"
        ],
        "plot_choices": "basic"
    },
    "path_generation_parameters":
    {
        "number_of_paths" : 2,
        "beta" : -1,
        "visualize_paths" : True
    }
}

def init_routerl_env(env_seed: int,
                     torch_seed: int,
                     env_params: dict,
                     dqn_kwargs: dict,
                     human_learning_episodes: int = 1) -> TrafficEnvironment:
    """
    1) Seeds SUMO/PettingZoo + PyTorch,
    2) Initializes the environment with the given parameters,
    3) Runs the human learning phase,
    4) Applies mutation and wraps mutated vehicles with DQN agents.

    Returns:
        env: TrafficEnvironment
        mutated_humans: dict str(agent_id) -> HumanAgent
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
    pre_mutation = env.all_agents.copy()
    env.mutation_odd_id_agents()

    # Set humans defaults
    for h in env.human_agents:
        h.default_action = 0

    # 4) Assign DQN to mutated vehicles
    machines = env.machine_agents.copy()
    free_flows = env.get_free_flow_times()
    mutated_humans = dict()
    for m in machines:
        for h in pre_mutation:
            if m.id == h.id:
                mutated_humans[str(m.id)] = h
                break

    for h_id, h in mutated_humans.items():
        initial_knowledge = free_flows[(h.origin, h.destination)]
        initial_knowledge = [0, 0]
        mutated_humans[h_id].model = DQN(
            state_size=3,
            action_space_size=len(initial_knowledge),
            **dqn_kwargs
        )

    return env, mutated_humans

def train_dqn(env_seed: int,
              torch_seed: int,
              env_params: dict,
              dqn_kwargs: dict) -> float:
    """
    Initializes env + agents, runs TRAINING_EPISODES of learning,
    then TESTING_EPISODES to measure avg travel time.
    Returns: avg_travel_time (float)
    """

    # Initialize environment and agents
    env, mutated_humans = init_routerl_env(
        env_seed=env_seed,
        torch_seed=torch_seed,
        env_params=env_params,
        dqn_kwargs=dqn_kwargs,
        human_learning_episodes=HUMAN_LEARNING_EPISODES
    )

    total_reward = 0.0

    # Training phase
    for _ in range(TRAINING_EPISODES):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                obs = [{kc.AGENT_ID : int(agent), kc.TRAVEL_TIME : -reward}]
                last_action = mutated_humans[agent].last_action
                mutated_humans[agent].learn(last_action, obs)
                action = None
            else:
                action = mutated_humans[agent].act(observation)
            env.step(action)

    # Testing phase
    travel_times = []
    for _ in range(TESTING_EPISODES):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = mutated_humans[agent].act(observation)
            env.step(action)
            # The reward is minus the travel time in this case.
            total_reward -= reward
            travel_times.append(-reward)

    avg_travel_time = np.mean(travel_times)
    std_travel_time = np.std(travel_times)

    # Clean up
    env.stop_simulation()

    return float(avg_travel_time),  float(std_travel_time)

def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna optimization.
    It defines the hyperparameters to tune and returns the average travel time.
    """
    dqn_kwargs = {
        param: trial.suggest_categorical(param, ALL_PARAM_GRIDS[param])
        for param in selected
    }

    run = wandb.init(
        project="DQN_Hyperparameter_Tuning_TEST",
        name=f"trial-{trial.number}",
        reinit=True,
        config=dqn_kwargs
    )

    wandb.log(dqn_kwargs, commit=False)

    times = []
    for env_seed in ENV_SEEDS:
        for torch_seed in TORCH_SEEDS:
            avg_time, test_std_time = train_dqn(env_seed, torch_seed, env_params, dqn_kwargs)
            wandb.log({
                "avg_travel_time": avg_time,
                "test_std_travel_time": test_std_time,
                "env_seed": env_seed,
                "torch_seed": torch_seed
            })
            times.append(avg_time)

    mean_time, std_time = float(np.mean(times)), float(np.std(times))
    wandb.log({"mean_travel_time": mean_time, "std_travel_time": std_time})
    run.finish()
    return mean_time

def main():
    # Set up Optuna grid search
    search_space = {k: ALL_PARAM_GRIDS[k] for k in selected}

    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Run the optimization
    study.optimize(objective, n_trials=int(np.prod([len(search_space[k]) for k in search_space])))

    # Log Optuna built-in plots to Weights & Biases
    wandb.init(
      project="DQN_Hyperparameter_Tuning",
      name="optuna_summary",
      reinit=True
    )

    wandb.log({
        "optuna_history": wandb.Plotly(plot_optimization_history(study)),
        "param_importances": wandb.Plotly(plot_param_importances(study))
    })
    wandb.finish()

if __name__ == "__main__":
    main()