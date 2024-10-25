# sklearn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit

# external
from skbio.stats.ordination import pcoa
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def init_config() -> Tuple[DictConfig, Dict[str, Callable]]:
    """
    Returns the configuration and score tracker for the kNN model.
    """
    logger.info("Initializing configuration...")
    cfg = DictConfig(
        {
            "data": {
                "scasim_matrix_path": "/home/shared/onestop_scasim_distances/distances_duration_norm.csv",
                "labels_path": "/home/shared/onestop_scasim_distances/distances_duration_norm_labels.csv",
                "item_columns": ["batch", "article_id", "paragraph_id"],
            },
            "validation": {
                "validation_order": ["new_item", "new_subject", "new_item_and_subject"],
                "fold_indices": [0, 2, 4, 6, 8],
                "num_folds": 10,
                "validation_metric": "accuracy",
            },
            "task": {
                # "target_column": "has_preview",
                # "pos_label": "Hunting",  #! corresponds to the target_column.
                "target_column": "level",
                "pos_label": "Adv",
            },
            "model": {
                "knn": {
                    "search_space": {
                        "n_neighbors": [3],
                        "weights": ["uniform"],
                        "metric": ["l2"],
                    },
                    "dim_reduction": {
                        # "sklearn.decomposition.PCA": [2, 4, 8, 16, 32],
                        "sklearn.decomposition.NMF": [2],
                        # "none": {},
                    },
                },
                "svm": {
                    "search_space": {
                        "C": [1],
                        "kernel": ["linear", "rbf"],
                        "degree": [3, 5],
                        "gamma": ["auto"],
                    },
                    "dim_reduction": {
                        # "sklearn.decomposition.PCA": [2, 4, 8, 16, 32],
                        "sklearn.decomposition.NMF": [128],
                        # "none": {},
                    },
                },
            },
            "output": {
                "output_dir": rf"{os.path.dirname}/outputs",
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
    }

    assert cfg.validation.validation_metric in score_tracker.keys(), print(
        f"Validation metric {cfg.validation.validation_metric} is not in the score tracker: {score_tracker.keys()}"
    )

    return cfg, score_tracker


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

        if dim_reduction_layer is None:
            model_space[f"{dim_reduction_name} -> {kNN_name}"] = make_pipeline(
                kNN_layer
            )
        else:
            model_space[f"{dim_reduction_name} -> {kNN_name}"] = make_pipeline(
                dim_reduction_layer, kNN_layer
            )

    return model_space


def main() -> None:
    cfg, score_tracker = init_config()
    model_space = init_model_space(cfg)
    logger.info(f"Model search space: {len(model_space)} models")

    logger.info("Loading data...")
    scasim_dist_matrix, labels = (
        pd.read_csv(cfg.data.scasim_matrix_path),
        pd.read_csv(cfg.data.labels_path),
    )

    scasim_dist_matrix, labels = process_data(cfg, scasim_dist_matrix, labels)
    labels[cfg.task.target_column] = (
        labels[cfg.task.target_column] == cfg.task.pos_label
    ).astype(int)
    grouped = labels.groupby(list(cfg.data.item_columns))
    items_labels = [group for _, group in grouped]

    # log summary of number of examples per item
    logger.info(f"Number of items: {len(items_labels)}")
    logger.info(
        f"Number of examples per item: {np.mean([len(item) for item in items_labels]):.2f} +/- {np.std([len(item) for item in items_labels]):.2f}"
    )

    score_object = {
        model_name: {
            "train": {score_name: [] for score_name in score_tracker.keys()},
            "val": {score_name: [] for score_name in score_tracker.keys()},
            "test": {score_name: [] for score_name in score_tracker.keys()},
        }
        for model_name in model_space.keys()
    }  # * list for different items

    for i, (model_name, model_pipline) in tqdm(
        enumerate(model_space.items()),
        desc="Model search space",
        position=0,
        leave=True,
        total=len(model_space),
    ):
        for j, item_labels in tqdm(
            enumerate(items_labels),
            desc="Items",
            position=1,
            leave=False,
            total=len(items_labels),
        ):
            train_test_sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.4, random_state=int(str(i) + str(j))
            )
            train_idx, valtest_idx = next(
                train_test_sss.split(item_labels, item_labels[cfg.task.target_column])
            )
            train_set = item_labels.iloc[train_idx]
            valtest_set = item_labels.iloc[valtest_idx]

            val_test_sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.5, random_state=int(str(i) + str(j))
            )
            val_idx, test_idx = next(
                val_test_sss.split(valtest_set, valtest_set[cfg.task.target_column])
            )
            val_set = valtest_set.iloc[val_idx]
            test_set = valtest_set.iloc[test_idx]

            X_train = scasim_dist_matrix.loc[
                train_set["unique_id"], train_set["unique_id"]
            ].values
            y_train = train_set[cfg.task.target_column].values

            X_val = scasim_dist_matrix.loc[
                val_set["unique_id"], train_set["unique_id"]
            ].values
            y_val = val_set[cfg.task.target_column].values

            X_test = scasim_dist_matrix.loc[
                test_set["unique_id"], train_set["unique_id"]
            ].values
            y_test = test_set[cfg.task.target_column].values

            model_pipline.fit(X_train, y_train)
            train_pred = model_pipline.predict(X_train)
            val_pred = model_pipline.predict(X_val)
            test_pred = model_pipline.predict(X_test)

            for score_name, score_func in score_tracker.items():
                train_score = score_func(y_train, train_pred)
                val_score = score_func(y_val, val_pred)
                test_score = score_func(y_test, test_pred)

                score_object[model_name]["train"][score_name].append(train_score)
                score_object[model_name]["val"][score_name].append(val_score)
                score_object[model_name]["test"][score_name].append(test_score)

    # save the results
    os.makedirs(cfg.output.output_dir, exist_ok=True)
    output_filename = f"raw_scores.{os.path.basename(__file__).replace('.py', '')}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"

    with open(os.path.join(cfg.output.output_dir, output_filename), "w") as f:
        logger.info(f"Saving raw scores to {output_filename}")
        json.dump(score_object, f, cls=NumpyEncoder)

    # log the results
    ## take the model that performed the best on the validation set averaged over items
    best_model = max(
        model_space.keys(),  # model names
        key=lambda model_name: np.mean(
            score_object[model_name]["val"][cfg.validation.validation_metric]
        ),
    )

    logger.info(f"Best model: {best_model}")

    # log table of metrics for the best model on validation set and test set
    # mean and std for each metric
    logger.info("\nTrain set ----------------")
    for score_name in score_tracker.keys():
        mean = np.mean(score_object[best_model]["train"][score_name])
        std = np.std(score_object[best_model]["train"][score_name])
        logger.info(f"{score_name}: {mean:.3f} +/- {std:.3f}")

    logger.info("\nValidation set -----------")
    for score_name in score_tracker.keys():
        mean = np.mean(score_object[best_model]["val"][score_name])
        std = np.std(score_object[best_model]["val"][score_name])
        logger.info(f"{score_name}: {mean:.3f} +/- {std:.3f}")

    logger.info("\nTest set -----------------")
    for score_name in score_tracker.keys():
        mean = np.mean(score_object[best_model]["test"][score_name])
        std = np.std(score_object[best_model]["test"][score_name])
        logger.info(f"{score_name}: {mean:.3f} +/- {std:.3f}")


def visualise():
    cfg, _ = init_config()

    logger.info("Loading data...")
    scasim_dist_matrix, labels = (
        pd.read_csv(cfg.data.scasim_matrix_path),
        pd.read_csv(cfg.data.labels_path),
    )

    scasim_dist_matrix, labels = process_data(cfg, scasim_dist_matrix, labels)
    labels[cfg.task.target_column] = (
        labels[cfg.task.target_column] == cfg.task.pos_label
    ).astype(int)

    if list(cfg.data.item_columns) == []:
        items_labels = [labels]
        items_names = ["all"]
    else:
        grouped = labels.groupby(list(cfg.data.item_columns))
        items_labels = [group for _, group in grouped]
        items_names = [gn for gn, _ in grouped]

    # log summary of number of examples per item
    logger.info(f"Number of items: {len(items_labels)}")
    logger.info(
        f"Number of examples per item: {np.mean([len(item) for item in items_labels]):.2f} +/- {np.std([len(item) for item in items_labels]):.2f}"
    )

    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2)

    for i, item_labels in tqdm(
        enumerate(items_labels),
        desc="Items",
        position=0,
        leave=True,
        total=len(items_labels),
    ):
        # pcoa
        X_i_2d = pcoa(
            scasim_dist_matrix.loc[item_labels["unique_id"], item_labels["unique_id"]],
            number_of_dimensions=2,
        ).samples
        fig.add_trace(
            go.Scatter(
                x=X_i_2d["PC1"].to_list(),
                y=X_i_2d["PC2"].to_list(),
                mode="markers",
                marker=dict(
                    size=12,
                    color=(
                        item_labels[cfg.task.target_column] == cfg.task.pos_label
                    ).astype(int),
                    line_width=1,
                ),
                name=f"{tuple(cfg.data.item_columns)}={items_names[i]}",
                visible=(i == 0),  # only the first scatter plot is visible
            ),
            row=1,
            col=1,  # this trace goes in the first subplot
        )

        # order item_labels by has_preview
        item_labels = item_labels.sort_values(cfg.task.target_column)

        X_i = scasim_dist_matrix.loc[
            item_labels["unique_id"], item_labels["unique_id"]
        ].values
        # convert to similarity
        X_i = 1 - X_i
        # plot X_i as a heatmap
        fig.add_trace(
            go.Heatmap(
                z=X_i,
                name=f"{tuple(cfg.data.item_columns)}={items_names[i]}",
                zmin=0,
                zmax=1,
                visible=(i == 0),  # only the first heatmap is visible
            ),
            row=1,
            col=2,  # this trace goes in the second subplot
        )

    # add slider
    steps = []
    for i in range(len(items_labels)):
        step_visibility = [False] * len(items_labels) * 2  # initialize all to False
        step_visibility[i * 2] = True  # set scatter plot of i-th item to True
        step_visibility[i * 2 + 1] = True  # set heatmap of i-th item to True
        step = dict(
            method="restyle",
            args=["visible", step_visibility],
            label=f"{tuple(cfg.data.item_columns)}={items_names[i]}",
        )
        steps.append(step)

    sliders = [dict(active=0, currentvalue={"prefix": "Item: "}, steps=steps)]

    fig.update_layout(sliders=sliders)

    # save the plot
    output_filename = f"visualisation.{os.path.basename(__file__).replace('.py', '')}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.html"
    logger.info(f"Saving visualisation to {output_filename}")
    fig.write_html(output_filename)

    fig.show()


if __name__ == "__main__":
    main()
