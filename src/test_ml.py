"""
__summary__:
    Given a list of experiments, where each experiment is a dictionary
    that maps fold_idx to a completed w&b sweep, this script ought to
    1. take the best hyperparameters from each fold.
    2. fit the model on that fold.
    3. save test predictions to file.
"""

from dataclasses import dataclass, replace
from typing import Callable

from tap import Tap
import wandb
from src.configs.constants import MLModelNames, Scaler
from src.configs.data_args import DataArgs
from src.configs.data_path_args import DataPathArgs
from src.configs.main_config import MLArgs, get_model_ml, move_target_column_to_end
from src import datamodule
from src.configs.model_args.base_model_args_ml import BaseMLModelArgs
from src.configs.model_args.ml_model_specific_args.DummyClassifierMLArgs import (
    DummyClassifierCondPredMLArgs,
)
from src.configs.model_args.ml_model_specific_args.LogisticRegressionMLArgs import (
    LogisticRegressionCondPredDianeMLArgs,
    LogisticRegressionCondPredAvgDwellTimeMLArgs,
    LogisticRegressionCondPredReadingSpdMLArgs,
)
from src.configs.trainer_args_ml import BaseMLTrainerArgs

import torch
import lightning_fabric as lf
import pandas as pd
import os
from tqdm import tqdm as my_beloved_slave
from pathlib import Path


class HyperArgs(Tap):
    wandb_project: str = "task-decoding"  # Name of the wandb project to log to.
    wandb_entity: str = "lacc-lab"  # Name of the wandb entity to log to.


@dataclass
class Sweep:
    sweep_id: str
    cfg_of_best: dict
    fold_index: int | None = None


@dataclass
class Experiment:
    experiment_name: str
    sweeps: list[Sweep]
    model_name: MLModelNames
    model_args_class: Callable


experiments: list[Experiment] = [
    # # KNN
    # Experiment(
    #     experiment_name="knn_emnlp_g_is_corr",
    #     model_name=MLModelNames.KNN,
    #     model_args_class=KNearestNeighborsIsCorrectPredEMNLPMLArgs,
    #     sweeps=[
    #         Sweep(
    #             sweep_id="vdzrula4",
    #             cfg_of_best={},
    #         ),
    #         Sweep(sweep_id="pbjy8dq5", cfg_of_best={}),
    #         Sweep(sweep_id="9d00jr78", cfg_of_best={}),
    #         Sweep(sweep_id="rveonob6", cfg_of_best={}),
    #         Sweep(sweep_id="3amyf66e", cfg_of_best={}),
    #         Sweep(sweep_id="65dpg1g4", cfg_of_best={}),
    #         Sweep(sweep_id="x65n56gm", cfg_of_best={}),
    #         Sweep(sweep_id="15a2s5xh", cfg_of_best={}),
    #         Sweep(sweep_id="e51aiwz0", cfg_of_best={}),
    #         Sweep(sweep_id="x2t6qfz6", cfg_of_best={}),
    #     ],
    # ),
    # ? Condition Prediction
    Experiment(
        experiment_name="logreg_cond_pred_ReadingSpeed",
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_args_class=LogisticRegressionCondPredReadingSpdMLArgs,
        sweeps=[
            Sweep(sweep_id="btvkfmqh", cfg_of_best={}),
            Sweep(sweep_id="l6hri8bb", cfg_of_best={}),
            Sweep(sweep_id="u18woe36", cfg_of_best={}),
            Sweep(sweep_id="n6u8un46", cfg_of_best={}),
            Sweep(sweep_id="o8e81pvi", cfg_of_best={}),
            Sweep(sweep_id="9vbiou7r", cfg_of_best={}),
            Sweep(sweep_id="isi0792n", cfg_of_best={}),
            Sweep(sweep_id="2byio50x", cfg_of_best={}),
            Sweep(sweep_id="ya3h691f", cfg_of_best={}),
            Sweep(sweep_id="hs7qnu7x", cfg_of_best={}),
        ],
    ),
]

# reverse order of sweeps in each experiment
for exp in experiments:
    exp.sweeps = exp.sweeps[::-1]


def get_best_run(wandb_api, sweep_path: str):
    # Get best run parameters
    sweep = wandb_api.sweep(sweep_path)
    best_run = (
        sweep.best_run()
    )  # when no metric is specified, the default is the sweep's goal metric
    return best_run.config


def checks(experiments: list[Experiment]) -> None:
    # check that the sweeps are unique
    sweep_ids = [sweep.sweep_id for exp in experiments for sweep in exp.sweeps]
    assert len(sweep_ids) == len(set(sweep_ids))


def main() -> None:
    checks(experiments)

    regime_names = [
        "new_item",
        "new_subject",
        "new_item_and_subject",
        # "all",
    ]  # This order is defined in the data module!

    lf.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    api = wandb.Api()

    hyper_args = HyperArgs().parse_args()

    # Get best hyperparameters for each fold
    for exp in experiments:
        group_level_metrics = []
        for sweep in my_beloved_slave(exp.sweeps):
            cfg_of_best = get_best_run(
                api,
                f"{hyper_args.wandb_entity}/{hyper_args.wandb_project}/{sweep.sweep_id}",
            )
            sweep.cfg_of_best = cfg_of_best
            sweep.fold_index = int(cfg_of_best["data"]["fold_index"])

            data_args = DataArgs(**sweep.cfg_of_best["data"])
            data_path_args = DataPathArgs(**sweep.cfg_of_best["data_path"])
            # for each att in data_path_args, apply Path() to it
            for att in data_path_args.__dict__.keys():
                setattr(data_path_args, att, Path(getattr(data_path_args, att)))
            trainer_args = BaseMLTrainerArgs(**sweep.cfg_of_best["trainer"])
            model_params_update = sweep.cfg_of_best["model"]["model_params"]
            sweep.cfg_of_best["model"].pop("model_params")
            model_args = exp.model_args_class(**sweep.cfg_of_best["model"])
            replace(model_args.model_params, **model_params_update)
            cfg = MLArgs(
                data=data_args,
                data_path=data_path_args,
                model=model_args,
                trainer=trainer_args,
            )
            cfg.data.normalization_type = Scaler.ROBUST_SCALER
            cfg = move_target_column_to_end(cfg=cfg)  # type: ignore

            dm = datamodule.ETDataModuleFast(cfg=cfg)
            dm.prepare_data()
            dm.setup(stage="fit")

            if cfg.model.model_params.class_weights is not None:
                class_weights = dm.train_dataset.ordered_label_counts["count"]
                cfg.model.model_params.class_weights = (
                    sum(class_weights) / class_weights
                ).tolist()
                print(f"Class weights: {cfg.model.model_params.class_weights}")

            assert isinstance(cfg.trainer, BaseMLTrainerArgs)
            assert isinstance(cfg.model, BaseMLModelArgs)
            model = get_model_ml(trainer_args=cfg.trainer, model_args=cfg.model)

            model.fit(dm=dm)

            dm.setup(stage="predict")  # creates val and test sets

            results = []
            for _, val_dataset in zip(regime_names, dm.val_datasets):
                results.append(model.predict(val_dataset))

            for _, test_dataset in zip(regime_names, dm.test_datasets):
                results.append(model.predict(test_dataset))

            for index, eval_type_results in enumerate(results):
                if index in [0, 1, 2]:
                    items, subjects, test_condition = zip(
                        *[
                            (i[5], i[6], i[10])  # type: ignore
                            for i in dm.val_datasets[index].ordered_key_list
                        ]  # type: ignore
                    )
                else:
                    items, subjects, test_condition = zip(
                        *[
                            (i[5], i[6], i[10])  # type: ignore
                            for i in dm.test_datasets[index % 3].ordered_key_list
                        ]  # type: ignore
                    )

                preds, probs, y_true, ordered_preds, ordered_probs, ordered_y_true = (
                    eval_type_results
                )

                # take probability of class#1
                probs = probs[:, 1]

                eval_regime = regime_names[index % 3]
                eval_type = "val" if index in [0, 1, 2] else "test"

                if preds.ndim == 1:
                    binary_labels = y_true
                    binary_preds = preds
                    binary_probs = probs
                else:
                    raise NotImplementedError

                df = pd.DataFrame(
                    {
                        "subjects": subjects,
                        "items": items,
                        "condition": test_condition,
                        "binary_label": binary_labels.numpy(),
                        "binary_prediction_prob": binary_probs.numpy(),
                        "label": y_true.numpy(),  # type: ignore
                        "prediction_prob": probs.numpy(),  # type: ignore
                        "eval_regime": eval_regime,
                        "eval_type": eval_type,
                        "fold_index": sweep.fold_index,
                        "ia_query": cfg.data.ia_query,
                    }
                )
                group_level_metrics.append(df)
        res = pd.concat(group_level_metrics)
        res["binary_prediction"] = (res["binary_prediction_prob"] > 0.5).astype(int)
        res["is_correct"] = (res["binary_label"] == res["binary_prediction"]).astype(
            int
        )
        res["level"] = res["items"].str.split("_").str[2]

        save_path = (
            f"ml_test_results/{exp.experiment_name}_trial_level_test_results.csv"
        )
        # create path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        res.to_csv(save_path)


if __name__ == "__main__":
    main()
