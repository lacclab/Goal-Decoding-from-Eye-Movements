# sklearn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline

# external
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from typing import Tuple, Dict, Callable
import importlib
from itertools import product
from tqdm import tqdm
import os
import datetime
import json
from plotly.subplots import make_subplots
from plotly import graph_objects as go

# local
from src.FoldSplitter import FoldSplitter

# logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def init_config() -> Tuple[DictConfig, Dict[str, Callable], FoldSplitter]:
    """
    Returns the configuration and score tracker for the kNN model.
    """
    logger.info("Initializing configuration...")
    cfg = DictConfig(
        {
            "data": {
                "scasim_matrix_path": "/home/shared/onestop_scasim_distances/distances_duration_norm.csv",
                "labels_path": "/home/shared/onestop_scasim_distances/distances_duration_norm_labels.csv",
                "item_columns": ["batch", "article_id"],
            },
            "validation": {
                "validation_order": ["new_item", "new_subject", "new_item_and_subject"],
                "fold_indices": [0, 2, 4, 6, 8],
                "num_folds": 10,
                "validation_metric": "accuracy",
                "report_metrics": ["accuracy", "f1", "%pos"],
            },
            "task": {
                "target_column": "has_preview",
                "pos_label": "Hunting",  #! corresponds to the target_column.
            },
            "model": {
                "knn": {
                    "search_space": {
                        "n_neighbors": [1, 3, 5, 7, 9],
                        "weights": ["distance", "uniform"],
                        "metric": ["precomputed", "l2"],
                    },
                    "dim_reduction": {
                        # "sklearn.decomposition.PCA": [2, 4, 8, 32],
                        "sklearn.decomposition.NMF": [2, 4, 8, 32, 128],
                        "none": {},
                    },
                },
                "svm": {
                    "search_space": {
                        "C": [0.1, 1, 10, 100],
                        "kernel": [
                            "linear",
                            "rbf",
                        ],
                        "degree": [2, 4, 8],
                        "gamma": ["auto", "scale"],
                    },
                    "dim_reduction": {
                        # "sklearn.decomposition.PCA": [2, 4, 8, 16, 32, 64, 128],
                        "sklearn.decomposition.NMF": [2, 4, 8, 32, 128],
                        # "none": {},
                    },
                },
            },
            "output": {
                "output_dir": os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "outputs"
                ),
            },
        }
    )

    score_tracker = {
        "accuracy": lambda y, y_pred: np.mean(y == y_pred),
        "tn": lambda y, y_pred: np.sum((y == 0) & (y_pred == 0)),
        "fp": lambda y, y_pred: np.sum((y == 0) & (y_pred == 1)),
        "fn": lambda y, y_pred: np.sum((y == 1) & (y_pred == 0)),
        "tp": lambda y, y_pred: np.sum((y == 1) & (y_pred == 1)),
        "num_examples": lambda y, y_pred: len(y),
        "num_pos_examples": lambda y, y_pred: np.sum(y == 1),
        "num_neg_examples": lambda y, y_pred: np.sum(y == 0),
        "num_pos_pred": lambda y, y_pred: np.sum(y_pred == 1),
        "num_neg_pred": lambda y, y_pred: np.sum(y_pred == 0),
        "f1": lambda y, y_pred: 2
        * np.sum((y == 1) & (y_pred == 1))
        / (np.sum(y == 1) + np.sum(y_pred == 1)),
        "%pos": lambda y, y_pred: np.mean(y == 1),
    }

    fold_splitter = FoldSplitter(
        item_columns=list(cfg.data.item_columns),
        subject_column="subject_id",
        groupby_columns=[],
        num_folds=cfg.validation.num_folds,
    )

    assert cfg.validation.validation_metric in score_tracker.keys(), print(
        f"Validation metric {cfg.validation.validation_metric} is not in the score tracker: {score_tracker.keys()}"
    )

    assert np.all([rm in score_tracker.keys() for rm in cfg.validation.report_metrics])

    return cfg, score_tracker, fold_splitter


def load_data(cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the data from the given paths.
    """
    logger.info("Loading data...")
    scasim_matrix = pd.read_csv(cfg.data.scasim_matrix_path)
    labels = pd.read_csv(cfg.data.labels_path)
    return scasim_matrix, labels


def process_data(
    cfg: DictConfig, scasim_dist_matrix: pd.DataFrame, labels: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns the processed data. (scasim_dist_matrix, labels)
    """
    logger.info("Processing data...")
    #### labels:
    labels[["batch", "article_id", "level", "paragraph_id"]] = labels[
        "unique_paragraph_id"
    ].str.split("_", expand=True)
    labels["unique_id"] = (
        labels["has_preview"]
        + "_"
        + labels["subject_id"]
        + "_"
        + labels["unique_paragraph_id"]
        + "_"
        + labels["trial_id"].astype(str)
    )

    # order labels by trail_id
    labels = labels.sort_values("trial_id")

    scasim_dist_matrix = scasim_dist_matrix.drop(columns="Unnamed: 0")
    scasim_dist_matrix = scasim_dist_matrix.set_index(labels["unique_id"])
    scasim_dist_matrix.columns = labels["unique_id"]

    # remove reread examples
    labels = labels[labels["reread"] == 0]
    scasim_dist_matrix = scasim_dist_matrix.loc[
        labels["unique_id"], labels["unique_id"]
    ]

    return scasim_dist_matrix, labels


def init_model_space(cfg: DictConfig) -> Dict[str, Pipeline]:
    logger.info("Initializing model search space...")

    model_space = {}

    ## SVM
    # create dimension reduction options
    svm_dim_reductions_options = []
    for (
        dim_reduction_import_path,
        n_components_options,
    ) in cfg.model.svm.dim_reduction.items():
        if dim_reduction_import_path == "none":
            dim_reduction_layer = None
            svm_dim_reductions_options.append(("no_dim_reduction", dim_reduction_layer))
        else:
            module_name, class_name = dim_reduction_import_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            ClassObj = getattr(module, class_name)

            for n_components in n_components_options:
                dim_reduction_layer = ClassObj(n_components=n_components)
                svm_dim_reductions_options.append(
                    (f"type={class_name}.n_comp={n_components}", dim_reduction_layer)
                )

    # create SVM options
    svm_options = []
    for C, kernel, degree, gamma in product(*cfg.model.svm.search_space.values()):
        svm_layer = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
        svm_options.append(
            (f"SVM.C={C}.kernel={kernel}.degree={degree}.gamma={gamma}", svm_layer)
        )

    # create pipelines
    for (dim_reduction_name, dim_reduction_layer), (svm_name, svm_layer) in product(
        svm_dim_reductions_options, svm_options
    ):
        # skip impossible combinations:
        ## None

        if dim_reduction_layer is None:
            model_space[f"{dim_reduction_name} -> {svm_name}"] = make_pipeline(
                svm_layer
            )
        else:
            model_space[f"{dim_reduction_name} -> {svm_name}"] = make_pipeline(
                dim_reduction_layer, svm_layer
            )

    ## kNN
    # create dimension reduction options
    knn_dim_reductions_options = []
    for (
        dim_reduction_import_path,
        n_components_options,
    ) in cfg.model.knn.dim_reduction.items():
        if dim_reduction_import_path == "none":
            dim_reduction_layer = None
            knn_dim_reductions_options.append(("no_dim_reduction", dim_reduction_layer))
        else:
            module_name, class_name = dim_reduction_import_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            ClassObj = getattr(module, class_name)

            for n_components in n_components_options:
                dim_reduction_layer = ClassObj(n_components=n_components)
                knn_dim_reductions_options.append(
                    (f"type={class_name}.n_comp={n_components}", dim_reduction_layer)
                )

    # create kNN options
    kNN_options = []
    for n_neighbors, weights, metric in product(*cfg.model.knn.search_space.values()):
        kNN_layer = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, metric=metric
        )
        kNN_options.append(
            (f"kNN.n_neigh={n_neighbors}.w={weights}.metric={metric}", kNN_layer)
        )

    # create pipelines
    for (dim_reduction_name, dim_reduction_layer), (kNN_name, kNN_layer) in product(
        knn_dim_reductions_options, kNN_options
    ):
        # skip impossible combinations:
        ## if dim_reduction_layer is not None => metric must not be precomputed
        if dim_reduction_layer is not None and kNN_layer.metric == "precomputed":
            continue

        ## if dim_reduction_layer is None => metric must be precomputed
        if dim_reduction_layer is None and kNN_layer.metric != "precomputed":
            continue

        if dim_reduction_layer is None:
            model_space[f"{dim_reduction_name} -> {kNN_name}"] = make_pipeline(
                kNN_layer
            )
        else:
            model_space[f"{dim_reduction_name} -> {kNN_name}"] = make_pipeline(
                dim_reduction_layer, kNN_layer
            )

    return model_space


def main():
    cfg, score_tracker, fold_splitter = init_config()

    scasim_dist_matrix, labels = load_data(cfg)
    scasim_dist_matrix, labels = process_data(cfg, scasim_dist_matrix, labels)

    labels[cfg.task.target_column] = (
        labels[cfg.task.target_column] == cfg.task.pos_label
    ).astype(int)
    labels.reset_index(drop=True, inplace=True)

    fold_splitter.create_folds(group_keys=labels)

    model_space = init_model_space(cfg)
    logger.info(f"Number of models in the model space: {len(model_space)}")

    score_object = {
        fold_idx: {
            model_name: {
                "train": {score_name: [] for score_name in score_tracker.keys()},
                "val": {score_name: [] for score_name in score_tracker.keys()},
                "test": {score_name: [] for score_name in score_tracker.keys()},
            }
            for model_name in model_space.keys()
        }
        for fold_idx in cfg.validation.fold_indices
    }  # * list for different validation sets

    for i, (model_name, model_pipline) in tqdm(
        enumerate(model_space.items()),
        desc="Model search space",
        position=0,
        leave=True,
        total=len(model_space),
    ):
        for fold_idx in tqdm(
            cfg.validation.fold_indices, desc="Folds", position=1, leave=False
        ):
            train_labels, val_labels_list, test_labels_list = (
                fold_splitter._train_val_test_splits(
                    group_keys=labels, fold_index=fold_idx
                )
            )

            # train
            X_train = scasim_dist_matrix.loc[
                train_labels["unique_id"], train_labels["unique_id"]
            ]
            y_train = train_labels[cfg.task.target_column]

            logger.info(f"Training model: {model_name}")
            model_pipline.fit(X_train, y_train)

            logger.info(f"Val model {model_name} is trained on fold {fold_idx}")
            # train
            y_train_pred = model_pipline.predict(X_train)

            for score_name, score_func in score_tracker.items():
                score_object[fold_idx][model_name]["train"][score_name].append(
                    score_func(y_train, y_train_pred)
                )

            # val
            ## note that we have 3 different validation sets
            for val_type, val_labels in zip(
                cfg.validation.validation_order, val_labels_list
            ):
                X_val = scasim_dist_matrix.loc[
                    val_labels["unique_id"], train_labels["unique_id"]
                ]
                y_val = val_labels[cfg.task.target_column]

                y_val_pred = model_pipline.predict(X_val)

                for score_name, score_func in score_tracker.items():
                    score_object[fold_idx][model_name]["val"][score_name].append(
                        score_func(y_val, y_val_pred)
                    )

            # test
            ## note that we have 3 different validation sets
            for val_type, test_labels in zip(
                cfg.validation.validation_order, test_labels_list
            ):
                X_test = scasim_dist_matrix.loc[
                    test_labels["unique_id"], train_labels["unique_id"]
                ]
                y_test = test_labels[cfg.task.target_column]

                y_test_pred = model_pipline.predict(X_test)

                for score_name, score_func in score_tracker.items():
                    score_object[fold_idx][model_name]["test"][score_name].append(
                        score_func(y_test, y_test_pred)
                    )

    # save the scores
    output_dir = cfg.output.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_filename = (
        f"score_object_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    with open(os.path.join(output_dir, output_filename), "w") as f:
        json.dump(score_object, f, cls=NumpyEncoder)

    logger.info(f"Scores are saved to {output_dir}/{output_filename}")

    # inspect the scores
    # get the model that has the highest validation_metric averaged across the validation sets
    best_model_name_per_fold = {
        fold_idx: max(
            model_space.keys(),
            key=lambda model_name: np.mean(
                score_object[fold_idx][model_name]["val"][
                    cfg.validation.validation_metric
                ]
            ),
        )
        for fold_idx in cfg.validation.fold_indices
    }

    # print in a table like format the best model per fold
    logger.info("Best model per fold:")
    for fold_idx, best_model_name in best_model_name_per_fold.items():
        logger.info(f"Fold {fold_idx}: {best_model_name}")

    # create pandas dataframe with the scores
    # one for validation set and one for test set
    # columns are the validation types
    # row is the validation metric
    # average the scores across the folds
    for phase in ["val", "test"]:
        scores_df = pd.DataFrame(
            {
                val_type: {
                    val_metric: np.mean(
                        [
                            score_object[fold_idx][best_model_name_per_fold[fold_idx]][
                                phase
                            ][val_metric][val_type_idx]
                            for fold_idx in cfg.validation.fold_indices
                        ]
                    )
                    for val_metric in cfg.validation.report_metrics
                }
                for val_type_idx, val_type in enumerate(cfg.validation.validation_order)
            }
        )

        logger.info(f"Scores for {phase} set:")
        print(scores_df)

    # create plotly figure with 2 * 3 subplots
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            f"Phase: {phase}, Validation type: {val_type}"
            for phase, val_type in product(
                ["val", "test"], cfg.validation.validation_order
            )
        ],
    )

    for phase, val_type in product(["val", "test"], cfg.validation.validation_order):
        logger.info(f"Scores for {phase} set: {val_type}")
        val_type_idx = list(cfg.validation.validation_order).index(val_type)
        # plot confusion matrix averaged across the folds
        confusion_matrix = np.zeros((2, 2))
        for fold_idx in cfg.validation.fold_indices:
            best_model_name = best_model_name_per_fold[fold_idx]
            confusion_matrix += np.array(
                [
                    [
                        score_object[fold_idx][best_model_name][phase]["tn"][
                            val_type_idx
                        ],
                        score_object[fold_idx][best_model_name][phase]["fp"][
                            val_type_idx
                        ],
                    ],
                    [
                        score_object[fold_idx][best_model_name][phase]["fn"][
                            val_type_idx
                        ],
                        score_object[fold_idx][best_model_name][phase]["tp"][
                            val_type_idx
                        ],
                    ],
                ]
            )

        confusion_matrix = confusion_matrix / len(cfg.validation.fold_indices)

        # norm such that actuals sum to 1
        confusion_matrix[0] = confusion_matrix[0] / np.sum(confusion_matrix[0])
        confusion_matrix[1] = confusion_matrix[1] / np.sum(confusion_matrix[1])

        # plot confusion matrix
        # add 'Predicted' and 'Actual' labels
        x_labels = ["Predicted: 0", "Predicted: 1"]
        y_labels = ["Actual: 0", "Actual: 1"]

        fig.add_trace(
            go.Heatmap(
                z=confusion_matrix,
                x=x_labels,
                y=y_labels,
                showscale=True,
                colorscale="Greys",
                zmin=0,
                zmax=1,
            ),
            row=(phase == "val") + 1,
            col=val_type_idx + 1,
        )

    fig.update_layout(title_text="Confusion matrices")
    # save fig
    fig.write_html(
        os.path.join(
            output_dir,
            f"confusion_matrices_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html",
        )
    )
    fig.show()


if __name__ == "__main__":
    main()
