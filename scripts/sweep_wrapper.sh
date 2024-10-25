#!/bin/bash
search_algorithm=grid
folds="0" # 1 2 3 4 5 6 7 8 9" # separate by comma, e.g. "0,1,2,3,4"
wandb_project="task-decoding"
slurm=false
config_names=(
    # "postfusion_condpred"
    # "RoBERTeyeFixation_condpred"
    # "RoBERTeyeWords_condpred"
    # "plmas_condpred"
    # "plmasf_condpred"
    # "eyettention_condpred"
    # "beyelstm_condpred"
    "fse_condpred"
    # "haller_w_ff_condpred"
    # "fse_condpred"
)
run_cap=200
slurm_cpus=10
slurm_mem="75G"
for config_name in "${config_names[@]}"; do
    if [ "$slurm" = true ]; then
        python scripts/better_hyperparameters_sweep.py --config_name=$config_name --search_algorithm=$search_algorithm --folds $folds --wandb_project=$wandb_project --run_cap=$run_cap --create_slurm --slurm_cpus=$slurm_cpus --slurm_mem=$slurm_mem
    else
        python scripts/better_hyperparameters_sweep.py --config_name=$config_name --search_algorithm=$search_algorithm --folds $folds --wandb_project=$wandb_project --run_cap=$run_cap
    fi
done
