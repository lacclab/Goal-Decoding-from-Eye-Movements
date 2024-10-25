# Keep track of runs

## Sweeps

1. check that 'search_spaces_and_configs.search_space_by_model_name' has the correct hyperparameter search space for the model you wish to sweep.
This is currently per model class, not config.
2. add a new 'RunConfig' to 'search_spaces_and_configs.run_configs' with the desired hyperparameters.
3. Create the sweeps (the scripts will create an executable bash script for each sweep (fold_idx), which will be used to launch the wandb sweeps):
    3.1 run 'python scripts/better_hyperparameters_sweep.py --config-name <config_name> --run-cap <run_cap> --wandb-project <wandb_project> --wandb-entity <wandb_entity>'
    * notes:
    * * 'config_name' should be the key of the 'RunConfig' in 'run_configs'.
    * * 'config_name' is the only required argument.
    3.2 Alternatively run `sweep_wrapper.sh` (set slurm to true or false)
4. Run the sweep
    4.1 run the bash script ./<bash_script>.sh
    4.2 alternatively run the `run_generic_sweep_local.sh` wrapper for multiple local runs.
    alternatively for running on multiple lab servers run `create_sweep_config_csv.py` and then `run_sweep_configs_remote.py` after update the `sweep_configs.csv` file accordingly.
    4.3 alternatively for slurm runs run like `run_slurm_sweeps.sh`, updated with the relevant sweeps.
5. input the GPU index and the number of runs you want to launch on that GPU to start the sweeps.

## Cleanup

`python scripts/cleanup_models.py --keep_one_best --real_run`

From utils:

`bash server_sync/run_command_all_servers.sh "cd Cognitive-State-Decoding; python scripts/cleanup_models.py --keep_one_best --real_run"`

## rsyncs

* rsync from nlp18,19,20 to srv2:
Make sure syncer.sh is updated with paths and then `bash syncer.sh`.

`python scripts/cleanup_models.py --keep_one_best --real_run`

`rsync -avzP --append-verify emnlp24_outputs/outputs/ shubi@dgx-master.technion.ac.il:/rg/berzak_prj/shubi/Cognitive-State-Decoding/synced_outputs/`

`rsync -avzP --append-verify --include='*.ckpt' --exclude='*' shubi@dgx-master.technion.ac.il:/rg/berzak_prj/shubi/Cognitive-State-Decoding/outputs/+data=NoReread,+data_path=august06,+model=RoberteyeConcatCondPredFixationsArgs,+trainer=CondPredBase,trainer.wandb_job_type=hyperparameter_sweep_RoberteyeConcatCondPredFixationsArgs/ /data/home/shared/goal_decoding/+data=NoReread,+data_path=august06,+model=RoberteyeConcatCondPredFixationsArgs,+trainer=CondPredBase,trainer.wandb_job_type=hyperparameter_sweep_RoberteyeConcatCondPredFixationsArgs/`


* slurm_rsync/slurm_rsync.job

* test_results files from dgx to srv2:
`rsync -ravzP --mkpath --append-verify --include="*/" --include="*test_results.csv" --exclude='*' --prune-empty-dirs shubi@dgx-master.technion.ac.il:/rg/berzak_prj/shubi/Cognitive-State-Decoding/synced_outputs/ emnlp24_results/`

## Slurms


## Evals
