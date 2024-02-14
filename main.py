from environment import TrafficEnvironment
from keychain import Keychain as kc
from services import Trainer
from services import create_agent_objects
from services import confirm_env_variable
from services import get_json
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from pettingzoo.test import parallel_api_test
import numpy as np
import os
from gymnasium.spaces import Box, Discrete
import gymnasium as gym
from pettingzoo.test import test_save_obs
from stable_baselines3 import PPO
import supersuit as ss
from Sumo_controller import Sumo
import torch.nn as nn


import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

import pandas as pd
from typing import Dict, Any
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

N_EVALUATIONS = 10  # Number of evaluations during the training
N_TIMESTEPS = int(2e4)  # Training budget
N_EVAL_EPISODES = 10
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_STARTUP_TRIALS = 4  # Stop random sampling after N_STARTUP_TRIALS
N_TRIALS = 1000  # Maximum number of trials
N_JOBS = 1 # Number of jobs to run in parallel
TIMEOUT = int(60 * 15)  # 15 minutes



confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_json(kc.PARAMS_PATH)

"""
def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:

    # Discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True) ## maximum value for the gradient clipping

    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["tine", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])


    # Display true values
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("max_grad_norm", max_grad_norm)

    net_arch = [
        {"pi": [64], "vf": [64]}
        if net_arch == "tiny"
        else {"pi": [64, 64], "vf": [64, 64]}
    ]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]


    return {
        #"n_steps": n_steps,
        "gamma": gamma,
        "max_grad_norm": max_grad_norm,
        "learning_rate": learning_rate,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
    }

def objective(trial: optuna.Trial) -> float:
    
    env = TrafficEnvironment(params[kc.SIMULATION_PARAMETERS])

    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, 1, num_cpus=2, base_class="stable_baselines3")

    kwargs = {
        "policy": "MlpPolicy",
        "env": env
    }

    # 1. Sample hyperparameters and update the keyword arguments
    kwargs.update(sample_a2c_params(trial))

    # Create the RL model
    model = A2C(**kwargs)

    # 3. Create the `TrialEvalCallback` callback defined above that will periodically evaluate
    # and report the performance using `N_EVAL_EPISODES` every `EVAL_FREQ`
    # TrialEvalCallback signature:
    # TrialEvalCallback(eval_env, trial, n_eval_episodes, eval_freq, deterministic, verbose)
    eval_callback = TrialEvalCallback(env,
                                       trial,
                                        N_EVAL_EPISODES, 
                                        EVAL_FREQ, 
                                        deterministic=True)

    nan_encountered = False
    try:
        # Train the model
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        env.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    
    print("\n\n\nlast mean reward is: ", eval_callback.last_mean_reward, "\n\n\n")

    return eval_callback.last_mean_reward
    
class TrialEvalCallback(EvalCallback):
    

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True"""


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

    model.learn(total_timesteps=60)

    print(f"[SUCCESS] Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

def main():

    Sumo_sim=Sumo(params)
    Sumo_sim.Sumo_start()

    env = TrafficEnvironment(params[kc.SIMULATION_PARAMETERS])
    
    parallel_api_test(env, num_cycles=1_000_000)
    print("\n[SUCCESS] Passed parallel_api_test\n")

    env_kwargs = {}
    train_butterfly_supersuit(env, steps=100, seed=0, **env_kwargs) 

    # Select the sampler, can be random, TPESampler, CMAES, ...
    """sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )
    # Create the study and start the hyperparameter optimization
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")  

    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    study.trials_dataframe().to_csv("study_results_a2c_cartpole.csv")

    print(study.trials_dataframe())

    print("study is: ", study)
    
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()"""
    

    Sumo_sim.Sumo_stop()  

main()