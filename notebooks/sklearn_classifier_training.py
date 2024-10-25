import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import json
from collections import defaultdict

# call ../src/FoldSplitter.py
import sys

sys.path.append("src")

from FoldSplitter import FoldSplitter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import NMF, PCA


# ## Constants


DISTANCE_MATRIX_PATH = "data/raw/distances_duration_norm.csv"
LABELS_PATH = "data/raw/dis.csv"

VALIDATIONS_ORDER = ["new_item", "new_subject", "new_item_and_subject"]
FOLD_INDICES = [0, 2, 4, 6, 8]
NUM_FOLDS = 10
FOLD_SPLITTER = FoldSplitter(
    item_columns=["batch", "article_id"],
    subject_column="subject_id",
    groupby_columns=[],
    num_folds=NUM_FOLDS,
)

TARGET_COLUMN = "has_preview"
if TARGET_COLUMN == "has_preview":  # determine positive label
    POS_LABEL = "Hunting"
else:
    raise ValueError(f"Unsupported target column - {TARGET_COLUMN}")

SCORER = {
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


# ## Helper Functions


# ### Load Data


def load_distance_matrix(path=DISTANCE_MATRIX_PATH):
    distances_duration_norm = pd.read_csv(path)
    return distances_duration_norm


def load_labels(path=LABELS_PATH):
    labels = pd.read_csv(path)
    return labels


# ### Preprocess Data


def process_data(scasim_dist_matrix: pd.DataFrame, labels: pd.DataFrame):
    """
    Returns the processed data. (scasim_dist_matrix, labels)
    """
    scasim_dist_matrix = scasim_dist_matrix.copy()
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


# ### Side Functions


def element_wise_mean(list_of_lists):
    return [np.mean(x) for x in zip(*list_of_lists)]


def get_hyperparameters_combinations(
    hyperparameters: dict[str, list], filter_func=None
):
    """
    Get all possible hyperparameters combinations
    """
    keys = hyperparameters.keys()
    values = hyperparameters.values()

    ret = [defaultdict(lambda: None, dict(zip(keys, v))) for v in product(*values)]
    if filter_func is not None:
        ret = list(filter(filter_func, ret))

    return ret


def train_val_test_generator(labels, distances):
    for fold_index in FOLD_INDICES:
        train_keys, val_keys_list, test_keys_list = (
            FOLD_SPLITTER._train_val_test_splits(
                group_keys=labels, fold_index=fold_index
            )
        )

        # distances from train_keys to train_keys
        train_distances = distances.loc[train_keys.unique_id, train_keys.unique_id]

        # distances from val_keys to train_keys
        val_list_distances = [
            distances.loc[val_keys.unique_id, train_keys.unique_id]
            for val_keys in val_keys_list
        ]

        # distances from test_keys to train_keys
        test_list_distances = [
            distances.loc[test_keys.unique_id, train_keys.unique_id]
            for test_keys in test_keys_list
        ]

        yield (
            fold_index,
            train_distances,
            val_list_distances,
            test_list_distances,
            train_keys,
            val_keys_list,
            test_keys_list,
        )


def get_clasifier_no_dr(hyperparameters):
    if hyperparameters["classifier"] == "knn":
        clf = make_pipeline(
            KNeighborsClassifier(
                n_neighbors=hyperparameters["n_neighbors"],
                weights=hyperparameters["weights"],
                metric=hyperparameters["metric"],
            )
        )
        return clf


def get_classifier(hyperparameters):
    """return none if hyperparameters are invalid"""
    if hyperparameters["classifier"] == "knn":
        if (
            hyperparameters["dim_reduction"] is not None
            and hyperparameters["metric"] == "precomputed"
        ):
            return None
        if (
            hyperparameters["dim_reduction"] is None
            and hyperparameters["metric"] != "precomputed"
        ):  # this is for the case where we use the Scasim precomputed matrix (without dimensionality reduction)
            # and want to use it as a feature matrix instead of a distance matrix
            pass  #! change for return None if we dont want to allow this option
        clf = make_pipeline(
            KNeighborsClassifier(
                n_neighbors=hyperparameters["n_neighbors"],
                weights=hyperparameters["weights"],
                metric=hyperparameters["metric"],
            )
        )
        ## add dimensionality reduction
        if hyperparameters["dim_reduction"] is None:
            pass
        elif hyperparameters["dim_reduction"] == "pca":
            clf.steps.insert(
                0, ("pca", PCA(n_components=hyperparameters["n_components"]))
            )
        elif hyperparameters["dim_reduction"] == "nmf":
            clf.steps.insert(
                0, ("nmf", NMF(n_components=hyperparameters["n_components"]))
            )
        else:
            raise ValueError(
                f"Unsupported dimensionality reduction - {hyperparameters['dim_reduction']}"
            )

        return clf

    elif hyperparameters["classifier"] == "dummy":
        return DummyClassifier(strategy=hyperparameters["strategy"])
    else:
        return None


# ### Training


def _train_clf(
    clf,
    distances,
    labels,
    fold_index,
    train_distances,
    val_list_distances,
    test_list_distances,
    train_keys,
    val_keys_list,
    test_keys_list,
):
    labels_with_unique_id_as_index = labels.set_index("unique_id")

    clf.fit(
        train_distances,
        labels_with_unique_id_as_index.loc[train_distances.index, TARGET_COLUMN],
    )

    # predict val and test
    val_preds = [clf.predict(val_distances) for val_distances in val_list_distances]
    test_preds = [clf.predict(test_distances) for test_distances in test_list_distances]

    # calculate scores
    val_scores = {
        scorer_name: [
            scorer(
                labels_with_unique_id_as_index.loc[val_keys.unique_id, TARGET_COLUMN],
                val_pred,
            )
            for val_keys, val_pred in zip(val_keys_list, val_preds)
        ]
        for scorer_name, scorer in SCORER.items()
    }

    test_scores = {
        scorer_name: [
            scorer(
                labels_with_unique_id_as_index.loc[test_keys.unique_id, TARGET_COLUMN],
                test_pred,
            )
            for test_keys, test_pred in zip(test_keys_list, test_preds)
        ]
        for scorer_name, scorer in SCORER.items()
    }

    return val_scores, test_scores


def train_clf(dim_red_clfs, labels, distances):
    """
    @param hyperparameters: dict (name, values_list)
    """
    # print number of classifiers and hyperparameters
    for i, (dim_red, clfs) in enumerate(dim_red_clfs.items()):
        print(f"Classifier {i}:\n{dim_red}")
        for clf in clfs:
            print("\t" + str(clf))

    scores = {}
    for dim_red, clfs in tqdm(dim_red_clfs.items(), desc="Outer loop"):
        for (
            fold_index,
            train_distances,
            val_list_distances,
            test_list_distances,
            train_keys,
            val_keys_list,
            test_keys_list,
        ) in train_val_test_generator(labels, distances):
            # save index
            trian_index = train_distances.index
            val_index = [val_distances.index for val_distances in val_list_distances]
            test_index = [
                test_distances.index for test_distances in test_list_distances
            ]

            # perform dimensionality reduction once
            if dim_red is not None:
                train_distances = dim_red.fit_transform(train_distances)
                val_list_distances = [
                    dim_red.transform(val_distances)
                    for val_distances in val_list_distances
                ]
                test_list_distances = [
                    dim_red.transform(test_distances)
                    for test_distances in test_list_distances
                ]

                # restore index
                train_distances = pd.DataFrame(train_distances, index=trian_index)
                val_list_distances = [
                    pd.DataFrame(val_distances, index=val_index[i])
                    for i, val_distances in enumerate(val_list_distances)
                ]
                test_list_distances = [
                    pd.DataFrame(test_distances, index=test_index[i])
                    for i, test_distances in enumerate(test_list_distances)
                ]

            for clf in tqdm(
                clfs, desc=f"Dimensionality reduction {dim_red} on fold {fold_index}"
            ):
                print(f"Training classifier {clf} on fold {fold_index}...")

                # train and evaluate classifier
                val_scores, test_scores = _train_clf(
                    clf,
                    distances,
                    labels,
                    fold_index,
                    train_distances,
                    val_list_distances,
                    test_list_distances,
                    train_keys,
                    val_keys_list,
                    test_keys_list,
                )

                # save scores
                scores[fold_index] = scores.get(fold_index, {})  # default dict
                scores[fold_index][str(dim_red) + str(clf)] = {
                    "val": val_scores,
                    "test": test_scores,
                }

    return scores


def save_scores(scores, path):
    with open(path, "w") as f:
        json.dump(scores, f)


# ### Interperate Scores


def interprate_scores(scores):
    # interpret results

    ## for each fold, get the test scores for the best hyperparameters
    ## where the best hyperparameters are the ones that maximize the average of the scores
    best_hyperparameters_per_fold = {}
    for fold_index in scores:
        best_hyperparameters_per_fold[fold_index] = max(
            scores[fold_index].items(),
            key=lambda x: np.mean(x[1]["val"]["accuracy"]),
        )

    ## get the average of the scores for the best hyperparameters for each fold and average over all folds
    val_scores = {
        scorer_name: element_wise_mean(
            [
                scores[fold_index][best_hyperparameters_per_fold[fold_index][0]]["val"][
                    scorer_name
                ]
                for fold_index in scores
            ]
        )
        for scorer_name in SCORER.keys()
    }
    test_scores = {
        scorer_name: element_wise_mean(
            [
                scores[fold_index][best_hyperparameters_per_fold[fold_index][0]][
                    "test"
                ][scorer_name]
                for fold_index in scores
            ]
        )
        for scorer_name in SCORER.keys()
    }

    return best_hyperparameters_per_fold, val_scores, test_scores


def print_inter_scores_pretty(pretext, scores):
    print(pretext)
    """
    scores is a dict with the following structure:
    {
        scorer_name: [score_new-item, score_new-subject, score_new-item-and-subject]
    }
    print it as a table where columns are the val_order and rows are the scorer_name
    """
    df = pd.DataFrame(scores)

    # transpose
    df = df.T

    df.index = scores.keys()
    df.columns = VALIDATIONS_ORDER
    print(df, end="\n\n")


# ## Training


print("Loading labels and distances...")
distances = load_distance_matrix()


# ### kNN


print("Preprocessing labels and distances...")
labels = load_labels()
processed_distances, labels = process_data(distances, labels)

FOLD_SPLITTER.create_folds(group_keys=labels)

hyperparameters = {
    "n_neighbors": [1, 2, 3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["l2"],
    "classifier": ["knn"],
}

hyperparameters_pc = hyperparameters.copy()
hyperparameters_pc["metric"] = ["precomputed"]

dim_red_and_clfs = {
    None: [
        get_clasifier_no_dr(hp)
        for hp in get_hyperparameters_combinations(hyperparameters_pc)
    ],
    NMF(n_components=100): [
        get_clasifier_no_dr(hp)
        for hp in get_hyperparameters_combinations(hyperparameters)
    ],
    NMF(n_components=200): [
        get_clasifier_no_dr(hp)
        for hp in get_hyperparameters_combinations(hyperparameters)
    ],
    NMF(n_components=500): [
        get_clasifier_no_dr(hp)
        for hp in get_hyperparameters_combinations(hyperparameters)
    ],
    PCA(n_components=100): [
        get_clasifier_no_dr(hp)
        for hp in get_hyperparameters_combinations(hyperparameters)
    ],
    PCA(n_components=200): [
        get_clasifier_no_dr(hp)
        for hp in get_hyperparameters_combinations(hyperparameters)
    ],
    PCA(n_components=500): [
        get_clasifier_no_dr(hp)
        for hp in get_hyperparameters_combinations(hyperparameters)
    ],
}

print("Training classifiers...")
scores = train_clf(dim_red_and_clfs, labels, processed_distances)


# save scores
save_scores(scores, "scores.json")

# interpret results
best_hyperparameters_per_fold, val_scores, test_scores = interprate_scores(scores)

# print results
print_inter_scores_pretty("Validation scores:", val_scores)
print_inter_scores_pretty("Test scores:", test_scores)
best_hp = {k: v[0] for k, v in best_hyperparameters_per_fold.items()}
for k, v in best_hp.items():
    print(f"Fold {k}: {v}")
