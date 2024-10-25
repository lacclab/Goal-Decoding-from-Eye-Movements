import os
import stat

import wandb
from src.configs.constants import MLModelNames
from scripts.run_wrapper import (
    MLModelOptions,
    MLTrainerOptions,
    DataOptions,
    DataPathOptions,
)
from typing import Literal, TypedDict
from tap import Tap

from scripts.search_spaces_and_configs_ml import run_configs, search_space_by_model_name
from pathlib import Path


class MLRunConfig(TypedDict):
    model_name: MLModelNames
    model_variant: MLModelOptions
    data_variant: DataOptions
    data_path: DataPathOptions
    trainer_variant: MLTrainerOptions
    add_feature_search_space: bool


class HyperArgs(Tap):
    config_name: str
    run_cap: int | None = (
        None  # Maximum number of runs to execute. Not in use when using grid search.
    )
    wandb_project: str = "default_project"  # Name of the wandb project to log to.
    wandb_entity: str = "lacc-lab"  # Name of the wandb entity to log to.
    folds: list[int] = [0]  # List of fold indices to run.
    gpu_count: int = 1  # Number of GPUs to use.
    search_algorithm: Literal["bayes", "grid", "random"] = (
        "grid"  # Search algorithm to use.
    )
    unified_script_for_all_folds: bool = (
        False  # Whether to create a single bash script for all folds.
    )


def create_sweep_configs(hyper_args: HyperArgs):
    cfg = run_configs[hyper_args.config_name]
    search_space = search_space_by_model_name[cfg.model_name]

    sweep_configs = [
        {
            "program": "src/train_ml.py",
            "method": hyper_args.search_algorithm,
            "metric": {
                "goal": "maximize",
                "name": "balanced_classless_accuracy/val_weighted_average",
            },
            "entity": hyper_args.wandb_entity,
            "project": hyper_args.wandb_project,
            "parameters": search_space,
            "command": [
                "${env}",
                "${interpreter}",
                "${program}",
                "${args_no_hypens}",
                f"+model={cfg.model_variant}",
                f"+data={cfg.data_variant}",
                f"+data_path={cfg.data_path}",
                f"+trainer={cfg.trainer_variant}",
                f"data.fold_index={fold_idx}",
                f"trainer.wandb_job_type={hyper_args.config_name}",
            ],
        }
        for fold_idx in hyper_args.folds
    ]

    # add run_cap if not grid search
    if hyper_args.search_algorithm != "grid":
        for sweep_cfg in sweep_configs:
            sweep_cfg["run_cap"] = hyper_args.run_cap

    return sweep_configs


def launch_sweeps(hyper_args, sweep_configs):
    sweep_ids = [
        wandb.sweep(
            cfg, entity=hyper_args.wandb_entity, project=hyper_args.wandb_project
        )
        for cfg in sweep_configs
    ]
    return sweep_ids


def create_bash_scripts(hyper_args, sweep_ids):
    # time_rn = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = Path("bash_sweep_ml")
    base_path.mkdir(parents=True, exist_ok=True)
    if hyper_args.unified_script_for_all_folds:
        filename = base_path / f"run.cfg={hyper_args.config_name}.sh"

        main_command = "; ".join(
            [
                f"CUDA_VISIBLE_DEVICES=${{GPU_NUM}} wandb agent {hyper_args.wandb_entity}/{hyper_args.wandb_project}/{sweep_id}"
                for sweep_id in sweep_ids
            ]
        )

        with open(filename, "w") as f:
            f.write(
                f"""#!/bin/bash

if ! command -v tmux &>/dev/null; then
    echo "tmux could not be found, please install tmux first."
    exit 1  
fi

source $HOME/miniforge3/etc/profile.d/conda.sh
source $HOME/miniforge3/etc/profile.d/mamba.sh
mamba activate decoding
cd $HOME/Cognitive-State-Decoding

GPU_NUM=$1
RUNS_ON_GPU=${{2:-1}}
for ((i=1; i<=RUNS_ON_GPU; i++)); do
    session_name="wandb-gpu${{GPU_NUM}}-dup${{i}}-unified-{sweep_ids[0]}"
    tmux new-session -d -s "${{session_name}}" "{main_command}"; tmux set-option -t "${{session_name}}" remain-on-exit on
    echo "Launched W&B agent for GPU ${{GPU_NUM}}, Dup ${{i}} in tmux session ${{session_name}}"
done
"""
            )
        os.chmod(filename, os.stat(filename).st_mode | stat.S_IEXEC)
        print(f"Created bash script: {filename}")
    else:
        for sweep_id, fold_idx in zip(sweep_ids, hyper_args.folds):
            filename = (
                base_path / f"run.fold={fold_idx}.cfg={hyper_args.config_name}.sh"
            )  # .time={time_rn}
            with open(filename, "w") as f:
                f.write(
                    f"""#!/bin/bash

if ! command -v tmux &>/dev/null; then
    echo "tmux could not be found, please install tmux first."
    exit 1  
fi

source $HOME/miniforge3/etc/profile.d/conda.sh
source $HOME/miniforge3/etc/profile.d/mamba.sh
mamba activate decoding
cd $HOME/Cognitive-State-Decoding

GPU_NUM=$1
RUNS_ON_GPU=${{2:-1}}
for ((i=1; i<=RUNS_ON_GPU; i++)); do
    session_name="wandb-gpu${{GPU_NUM}}-dup${{i}}-{sweep_id}"
    tmux new-session -d -s "${{session_name}}" "CUDA_VISIBLE_DEVICES=${{GPU_NUM}} wandb agent {hyper_args.wandb_entity}/{hyper_args.wandb_project}/{sweep_id}"; tmux set-option -t "${{session_name}}" remain-on-exit on
    echo "Launched W&B agent for GPU ${{GPU_NUM}}, Dup ${{i}} in tmux session ${{session_name}}"
done
"""
                )
            os.chmod(filename, os.stat(filename).st_mode | stat.S_IEXEC)
            print(f"Created bash script: {filename}")


def main():
    hyper_args = HyperArgs().parse_args()
    print(f"Hyper Args:\n{hyper_args}")

    sweep_configs = create_sweep_configs(hyper_args)
    sweep_ids = launch_sweeps(hyper_args, sweep_configs)
    create_bash_scripts(hyper_args, sweep_ids)


if __name__ == "__main__":
    main()
