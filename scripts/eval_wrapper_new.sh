paths=(
    "outputs/+data=NoReread,+data_path=august06,+model=FixSeqEncCondPredArgs,+trainer=CondPredBase,trainer.wandb_job_type=hyperparameter_sweep_FixSeqEncCondPredArgs"
    "outputs/+data=NoReread,+data_path=august06,+model=BEyeLSTMCondPredArgs,+trainer=CondPredBase,trainer.wandb_job_type=hyperparameter_sweep_BEyeLSTMCondPredArgs"
    "outputs/+data=NoReread,+data_path=august06,+model=EyettentionCondPredArgs,+trainer=CondPredBase,trainer.wandb_job_type=hyperparameter_sweep_EyettentionCondPredArgs"
    "outputs/+data=NoReread,+data_path=august06,+model=PLMASCondPredArgs,+trainer=CondPredBase,trainer.wandb_job_type=hyperparameter_sweep_PLMASCondPredArgs"
    "outputs/+data=NoReread,+data_path=august06,+model=PLMASfCondPredArgs,+trainer=CondPredBase,trainer.wandb_job_type=hyperparameter_sweep_PLMASfCondPredArgs"
)

for path in "${paths[@]}"; do
    CUDA_VISIBLE_DEVICES=7 python src/eval.py "eval_path=\"${path}\""
done