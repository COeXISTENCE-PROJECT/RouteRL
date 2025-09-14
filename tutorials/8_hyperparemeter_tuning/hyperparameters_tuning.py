import numpy as np
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import plotly.express as px
import plotly.graph_objects as go
import wandb
from routerl import Keychain as kc



def run_gridsearch(train_fn: callable, 
                   ALL_PARAM_GRIDS: dict[str, list], 
                   selected: list[str], 
                   env_params: dict, 
                   ENV_SEEDS: list[int], 
                   TORCH_SEEDS: list[int], 
                   TRAINING_EPISODES: int, 
                   HUMAN_LEARNING_EPISODES: int, 
                   project_name: str):
    """
    Generic grid search optimization loop using Optuna + Weights & Biases.

    Args:
        train_fn: callable(env_seed, torch_seed, env_params, model_kwargs) -> (avg_time, std_time)
        ALL_PARAM_GRIDS: dict[str, list] - search space for hyperparameters
        selected: list[str] - parameters to tune in the current run
        env_params: dict - environment configuration
        ENV_SEEDS: list[int] - environment seeds to use
        TORCH_SEEDS: list[int] - torch seeds to use
        TRAINING_EPISODES: int - number of training episodes
        HUMAN_LEARNING_EPISODES: int - number of human learning episodes
        project_name: str - wandb project name
    """

    # accumulators across trials (persist across objective calls)
    trial_aggregates = []   # list of {"trial": int, "mean": float, "std": float}
    trial_seed_avgs = []    # list of {"trial": int, "avg_time": float} used for boxplot of means


    def objective(trial: optuna.Trial):
        # Sample hyperparameters for this trial
        kwargs = {
            param: trial.suggest_categorical(param, ALL_PARAM_GRIDS[param])
            for param in selected
        }

        trial_folder = f"training_records/trial_{trial.number}"
        env_params["plotter_parameters"][kc.RECORDS_FOLDER] = trial_folder

        # Start a new W&B run for the trial
        run = wandb.init(
            project=project_name,
            name=f"trial-{trial.number}",
            reinit=True,
            config=kwargs
        )
        wandb.log(kwargs, commit=False)


        # collect raw episode values and per-seed summaries for this trial
        results = []
        for env_seed in ENV_SEEDS:
            for torch_seed in TORCH_SEEDS:
                avg_time = train_fn(env_seed, torch_seed, env_params, kwargs)

                entry = {
                    "seed_pair": f"env{env_seed}_torch{torch_seed}",
                    "env_seed": int(env_seed),
                    "torch_seed": int(torch_seed),
                    "avg_time": float(avg_time)
                }
                results.append(entry)

                # log per-seed scalar summary
                wandb.log({
                    "avg_travel_time": avg_time,
                    "env_seed": env_seed,
                    "torch_seed": torch_seed
                }, commit=True)


        # -------------------------
        # Aggregate bar with error: mean and std of avg_time across seeds for this trial
        # -------------------------
        avg_times = [r["avg_time"] for r in results]
        mean_time_across_seeds = float(np.mean(avg_times))
        std_time_across_seeds = float(np.std(avg_times))

        # append to global accumulators for cross-trial plots
        trial_aggregates.append({
            "trial": int(trial.number),
            "mean": mean_time_across_seeds,
            "std": std_time_across_seeds
        })

        # store per-seed means for boxplot-of-means later
        for a in avg_times:
            trial_seed_avgs.append({
                "trial": int(trial.number),
                "avg_time": float(a)
            })

        # log aggregated scalars for this trial
        wandb.log({
            "mean_travel_time": mean_time_across_seeds,
            "aggregate_std_travel_time": std_time_across_seeds
        })

        # -------------------------
        # Combined bar plot: all trials on a single figure (update every trial)
        # -------------------------
        # Build sorted arrays from trial_aggregates
        trial_aggregates_sorted = sorted(trial_aggregates, key=lambda x: x["trial"])
        num_trials = len(trial_aggregates_sorted)
        if num_trials > 0:
            xs_idx = list(range(num_trials))
            xs_labels = [f"trial-{t['trial']}" for t in trial_aggregates_sorted]
            ys = [t["mean"] for t in trial_aggregates_sorted]
            errs = [t["std"] for t in trial_aggregates_sorted]

            fig_interactive = go.Figure()
            for i, (y, err, label) in enumerate(zip(ys, errs, xs_labels)):
                fig_interactive.add_trace(
                    go.Bar(
                        x=[i],
                        y=[y],
                        error_y=dict(type="data", array=[err], visible=True),
                        name=label,
                        marker=dict(opacity=0.85),
                        hovertemplate=f"{label}<br>Mean: %{y:.3f}<br>Std: {err:.3f}<extra></extra>"
                    )
                )

            fig_interactive.update_layout(
                title="Mean Travel Time ± Std for all Trials (aggregated across seeds)",
                xaxis=dict(
                    title="Trial",
                    tickmode="array",
                    tickvals=xs_idx,
                    ticktext=xs_labels,
                    showline=True, linewidth=2, linecolor="black",
                    rangeslider=dict(visible=True)
                ),
                yaxis=dict(title="Mean Travel Time [s]", showline=True, linewidth=2, linecolor="black"),
                font=dict(size=12),
                bargap=0.2,
                legend=dict(traceorder="normal")
            )

            K = min(10, num_trials)
            # indices of top/bottom by mean (smaller mean is better)
            sorted_by_mean_idx = sorted(range(num_trials), key=lambda i: ys[i])
            top_k_idx = sorted_by_mean_idx[:K]
            bottom_k_idx = sorted_by_mean_idx[-K:]

            # helper to build visibility mask for a set of indices
            def visibility_mask(selected_indices):
                return [i in selected_indices for i in range(num_trials)]

            buttons = [
                dict(
                    label="All",
                    method="update",
                    args=[{"visible": [True] * num_trials},
                        {"title": "Mean Travel Time ± Std — All Trials"}]
                ),
                dict(
                    label=f"Top {K}",
                    method="update",
                    args=[{"visible": visibility_mask(set(top_k_idx))},
                        {"title": f"Mean Travel Time ± Std — Top {K} Trials (lowest mean)"}]
                ),
                dict(
                    label=f"Bottom {K}",
                    method="update",
                    args=[{"visible": visibility_mask(set(bottom_k_idx))},
                        {"title": f"Mean Travel Time ± Std — Bottom {K} Trials (highest mean)"}]
                ),
            ]

            fig_interactive.update_layout(
                updatemenus=[
                    dict(
                        active=0,
                        buttons=buttons,
                        x=0.0,
                        y=1.15,
                        xanchor="left",
                        yanchor="top"
                    )
                ]
            )

            # log interactive figure
            wandb.log({"aggregate_all_trials_interactive": wandb.Plotly(fig_interactive)})


        # -------------------------
        # Boxplot: distribution of mean travel times (mean per seed-combo) grouped by trial
        # -------------------------
        # trial_seed_avgs is a flat list of dicts {"trial":int, "avg_time":float}
        fig_means_box = px.box(
            trial_seed_avgs,
            x="trial",
            y="avg_time",
            title="Distribution of mean travel times across seeds (grouped by trial)",
            labels={"trial": "Trial", "avg_time": "Mean Travel Time [s]"},
            points="all"
        )
        fig_means_box.update_layout(
            font=dict(size=12),
            xaxis=dict(showline=True, linewidth=2, linecolor="black"),
            yaxis=dict(showline=True, linewidth=2, linecolor="black"),
        )
        wandb.log({"mean_travel_time_boxplot_by_trial": wandb.Plotly(fig_means_box)})

        run.finish()
        # objective: minimize mean travel time across seeds
        return mean_time_across_seeds

    # Define grid search space and run Optuna GridSampler
    search_space = {k: ALL_PARAM_GRIDS[k] for k in selected}
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    n_trials = int(np.prod([len(search_space[k]) for k in search_space]))

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Final Optuna summary plots
    wandb.init(project=project_name, name="optuna_summary", reinit=True)

    fig1 = plot_optimization_history(study)
    fig1.update_layout(
        title="Optimization History",
        xaxis_title="Trial Number",
        yaxis_title="Objective Value (Avg Travel Time)",
        font=dict(size=14),
        xaxis=dict(showline=True, linewidth=2, linecolor="black"),
        yaxis=dict(showline=True, linewidth=2, linecolor="black"),
    )
    wandb.log({"optuna_history": wandb.Plotly(fig1)})

    fig2 = plot_param_importances(study)
    fig2.update_layout(
        title="Hyperparameter Importances",
        xaxis_title="Importance",
        yaxis_title="Hyperparameters",
        font=dict(size=14),
        xaxis=dict(showline=True, linewidth=2, linecolor="black"),
        yaxis=dict(showline=True, linewidth=2, linecolor="black"),
    )
    wandb.log({"param_importances": wandb.Plotly(fig2)})

    rows = []
    for trial in study.trials:
        mean_time = trial.user_attrs.get("mean_time", None)
        std_time = trial.user_attrs.get("std_time", None)

        row = {
            "trial": trial.number,
            "mean_time": mean_time,
            "std_time": std_time,
        }
        row.update(trial.params)
        rows.append(row)

    columns = list(rows[0].keys())
    data = [[row[col] for col in columns] for row in rows]
    table = wandb.Table(data=data, columns=columns)

    wandb.log({"hyperparameters_summary": table})
    wandb.finish()