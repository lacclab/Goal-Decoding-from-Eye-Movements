import ast
import json
from collections import defaultdict
from pathlib import Path
from tkinter import font

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import process_onestop_sr_report.preprocessing as prp
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

models = {
    "Dummy": "DummyClassifierCondPredMLArgs",
    "LRAvgDWELL": "LogisticRegressionCondPredAvgDwellTimeMLArgs",
    # "ReadingSpeed": "LogisticRegressionCondPredReadingSpeedMLArgs",
    "LRDiane": "LogisticRegressionCondPredDianeMLArgs",
    "FSE": "FixSeqEncCondPredArgs",
    "RoBERTa-QEye-W": "RoberteyeConcatCondPredWordsArgs",
    "MAG": "MAGCondPredArgs",
    "PLMAS": "PLMASCondPredArgs",
    "Haller RNN": "PLMASfCondPredArgs",
    "BEyeLSTM": "BEyeLSTMCondPredArgs",
    "Eyettention": "EyettentionCondPredArgs",
    "RoBERTa-QEye-F": "RoberteyeConcatCondPredFixationsArgs",
    "PostFusion": "PostFusionCondPred",
}


def read_res(
    data,
    model,
    trainer,
    data_path,
    base_path=Path("/data/home/shubi/Cognitive-State-Decoding"),
    base_res_path="cross_validation_runs",
    wandb_job_type="cv",
    template="{}/+data={},+data_path={},+model={},+trainer={},trainer.wandb_job_type={}",
    on_error="raise",
) -> pd.DataFrame | None:
    file_path = (
        base_path
        / template.format(
            base_res_path, data, data_path, model, trainer, wandb_job_type
        )
        / "trial_level_test_results.csv"
    )
    try:
        res = pd.read_csv(file_path, index_col=0)
    except FileNotFoundError as e:
        print(f"File not found: {file_path}")
        if on_error == "raise":
            raise e
        else:
            return None
    return res


def convert_string_to_list(s: pd.Series) -> list[list[float]]:
    return s.apply(ast.literal_eval).tolist()


def raw_res_to_metric(
    res: pd.DataFrame,
) -> pd.DataFrame:
    res_df = defaultdict(list)
    grouped_res = res.groupby(["eval_regime", "fold_index"])
    for (eval_type, fold_index), group_data in grouped_res:
        labels = group_data["label"].tolist()

        preds = group_data["prediction_prob"]
        if preds.dtype == "object":
            preds = convert_string_to_list(preds)
            y_pred = np.array(preds).argmax(axis=1)
        else:
            y_pred = group_data["y_pred"]

        accuracy = accuracy_score(y_true=labels, y_pred=y_pred)
        res_df[eval_type].append(round(accuracy, 3))

    grouped_res = res.groupby("fold_index")
    for fold_index, group_data in grouped_res:
        all_acc = accuracy_score(
            y_true=group_data["label"], y_pred=group_data["y_pred"]
        )
        res_df["all"].append(all_acc)

    mode = "accuracy"
    df = pd.DataFrame(
        res_df.items(),
        columns=["Eval Regime", mode],
    ).set_index("Eval Regime")

    new_order = ["new_item", "new_subject", "new_item_and_subject", "all"]
    df = df.reindex(new_order, level="Eval Regime")

    return df


def aggregate_df(
    res_df: pd.DataFrame,
    metric_name: str,
    columns: list[str] = ["new_item", "new_subject", "new_item_and_subject", "all"],
) -> pd.DataFrame:
    res_df = res_df.copy()
    for col in columns:
        avg_col = f"Avg {metric_name} {col}"
        std_col = f"Std {metric_name} {col}"
        res_df[avg_col] = res_df[col].apply(np.mean)
        res_df[std_col] = res_df[col].apply(np.std)
        res_df[avg_col] = (100 * res_df[avg_col]).round(1)
        res_df[std_col] = (100 * res_df[std_col]).round(1)
        res_df[f"{metric_name} {col}".replace("_", " ").capitalize()] = res_df.apply(
            lambda x: f"{x[avg_col]} ± {x[std_col]}",
            axis=1,
            # lambda x: f"{x[avg_col]}", axis=1,
        )
        res_df = res_df.drop([col, avg_col, std_col], axis=1)
    return res_df


def plot_grouped_accuracy(
    res,
    group_var,
    remove_legend: bool = False,
    figsize=None,
    continuous: bool = False,
    xlim=None,
):
    # Compute the average accuracy and error bar for each level
    accuracy_by_x = (
        res.groupby([group_var, "fold_index", "eval_regime"])["is_correct"]
        .mean()
        .reset_index()
    )

    # Plot the average accuracy with error bars for each level
    if figsize:
        plt.figure(figsize=figsize)
    if continuous:
        sns.lineplot(data=accuracy_by_x, x=group_var, hue="eval_regime", y="is_correct")
    else:
        sns.barplot(data=accuracy_by_x, x="eval_regime", hue=group_var, y="is_correct")

    plt.xlabel(group_var)
    plt.ylabel("Average Accuracy")
    plt.title(f"Average Accuracy by {group_var}")

    if xlim:
        plt.xlim(xlim)

    if remove_legend:
        plt.legend().remove()

    plt.show()


def prepare_dataframes(base_path):
    # Load and prepare eyes data
    dataset_path = Path(
        "../ln_shared_data/onestop/processed/ia_data_enriched_360_17092024.csv"
    )
    eyes = prp.load_data(data_path=dataset_path)
    eyes.subject_id = eyes.subject_id.str.lower()
    eyes_numerical = (
        eyes[eyes["reread"] == 0]
        .groupby(["unique_paragraph_id", "subject_id"])
        .mean(numeric_only=True)
    )
    # Keep only non-numeric columns and the unique paragraph id and subject id
    eyes_categorical = (
        eyes[eyes["reread"] == 0]
        .drop_duplicates(["unique_paragraph_id", "subject_id", "list"])
        .drop(set(eyes_numerical.columns) - set(["list"]), axis=1)[
            ["unique_paragraph_id", "subject_id", "question", "abcd_answer", "list"]
        ]
    )
    eyes_numerical = eyes_numerical.reset_index()
    dwell_per_span = (
        eyes.loc[
            eyes["reread"] == 0,
            ["relative_to_aspan", "IA_DWELL_TIME", "unique_paragraph_id", "subject_id"],
        ]
        .groupby(["unique_paragraph_id", "subject_id", "relative_to_aspan"])
        .mean()
        .unstack()
        .reset_index()
    )
    dwell_per_span.columns = [
        "_".join(map(str, col)).strip("_") for col in dwell_per_span.columns.values
    ]
    dwell_per_span.fillna(0, inplace=True)
    eyes_numerical = eyes_numerical.merge(
        right=dwell_per_span, on=["unique_paragraph_id", "subject_id"], how="left"
    )
    # Load and prepare text data
    text = pd.read_csv(base_path / "all_dat_files_merged.tsv", sep="\t")
    unique_item_columns = ["batch", "article_id", "level", "paragraph_id"]
    text["unique_paragraph_id"] = (
        text[unique_item_columns].astype(str).apply("_".join, axis=1)
    )
    text = text.drop_duplicates(["unique_paragraph_id", "q_ind", "list"])
    pattern = r"(\d+), ?(\d+)"
    text[["aspan_ind_start", "aspan_ind_end"]] = text.aspan_inds.str.extract(
        pattern, expand=True
    ).astype(int)
    text["aspan_len"] = text["aspan_ind_end"] - text["aspan_ind_start"]
    text["paragraph_length"] = text["paragraph"].apply(lambda x: len(x.split()))

    # Load and prepare metadata
    metadata = pd.read_csv(base_path / "full_report.csv")
    technion_experimenters = ["Aya", "Liz", "Nethanella"]
    metadata["data_collection_site"] = (
        metadata["Experimenter"]
        .isin(technion_experimenters)
        .replace({True: "Technion", False: "MIT"})
    )
    metadata["RECORDING_SESSION_LABEL"] = metadata[
        "RECORDING_SESSION_LABEL"
    ].str.lower()

    # Load session summary
    session_summary = pd.read_csv(filepath_or_buffer=base_path / "session_summary.csv")

    # Load and prepare questionnaire data
    questionnaire_path = base_path / "questionnaire.json"
    with open(questionnaire_path, "r") as file:
        questionnaire = json.load(file)
    data = []
    for response in questionnaire:
        participant_id = response["Participant ID"].lower()
        reading_habits = sum(response["Reading habits in English"].values())
        data.append(
            {"Participant ID": participant_id, "Reading Habits": reading_habits}
        )
    reading_habits = pd.DataFrame(data)

    # Load readability metrics
    readability_metrics = pd.read_csv(
        filepath_or_buffer=base_path / "paragraphs_metrics.csv"
    )

    return (
        eyes_numerical,
        eyes_categorical,
        text,
        metadata,
        session_summary,
        reading_habits,
        readability_metrics,
    )


def merge_dataframes(
    res,
    eyes_numerical,
    eyes_categorical,
    text,
    metadata,
    session_summary,
    reading_habits,
    readability_metrics,
):
    n_rows = res.shape[0]
    res = res.merge(
        reading_habits,
        how="left",
        left_on="subject_id_without_list",
        right_on="Participant ID",
        suffixes=(None, "_reading_habits"),
    )
    res = res.merge(
        metadata[
            [
                "RECORDING_SESSION_LABEL",
                "data_collection_site",
                "education",
                "gender",
                "age",
                "comprehension_score_without_reread",
            ]
        ],
        how="left",
        left_on="subjects",
        right_on="RECORDING_SESSION_LABEL",
        suffixes=(None, "_metadata"),
    )
    res = res.merge(
        eyes_numerical,
        how="left",
        left_on=["items", "subjects"],
        right_on=["unique_paragraph_id", "subject_id"],
        suffixes=(None, "_eyes"),
    )
    res = res.merge(
        eyes_categorical,
        how="left",
        left_on=["items", "subjects"],
        right_on=["unique_paragraph_id", "subject_id"],
        suffixes=(None, "_eyes_cat"),
    )
    res = res.merge(
        readability_metrics,
        how="left",
        left_on="items",
        right_on="unique_paragraph_id",
        suffixes=(None, "_readability"),
    )
    res.list = res.list.astype(int)
    res.q_ind = res.q_ind.astype(int)
    res = res.merge(
        text,
        how="left",
        left_on=["items", "q_ind", "list"],
        right_on=["unique_paragraph_id", "q_ind", "list"],
        suffixes=(None, "_text"),
        validate="many_to_one",
    )

    res["aspan_start_rel"] = res["aspan_ind_start"] / res["paragraph_length"]

    assert n_rows == res.shape[0], "Number of rows changed"
    return res


def plot_average_roc_curves_with_error_bands(
    model_dfs,
    base_path="figures",
):
    # Get unique evaluation regimes from the first model's data
    first_model_name = next(iter(model_dfs))
    unique_eval_regimes = model_dfs[first_model_name]["eval_regime"].unique()
    num_regimes = len(unique_eval_regimes)

    # Create subplots
    fig, axes = plt.subplots(
        1, num_regimes, figsize=(6 * num_regimes, 6), sharex=True, sharey=True
    )

    # Iterate over each evaluation regime
    for i, regime in enumerate(unique_eval_regimes):
        ax = axes[i] if num_regimes > 1 else axes

        # Iterate over each model
        for model_name, model_res in model_dfs.items():
            # Filter the DataFrame for the current eval_regime
            regime_res = model_res[model_res["eval_regime"] == regime]
            unique_folds = regime_res["fold_index"].unique()
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for fold in unique_folds:
                # Filter the DataFrame for the current fold
                fold_res = regime_res[regime_res["fold_index"] == fold]

                # Compute the false positive rate, true positive rate, and thresholds
                fpr, tpr, _ = roc_curve(fold_res["label"], fold_res["prediction_prob"])

                # Compute the AUROC score
                auroc = roc_auc_score(fold_res["label"], fold_res["prediction_prob"])
                aucs.append(auroc)

                # Interpolate the TPRs to the mean FPRs
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

            # Compute mean and standard deviation of TPRs and AUROC values
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)

            model_name_mapping = {
                "label": "Ground Truth Label",
                "LRAvgDWELL": "Reading Time",
                "A_LRAvgDWELL": "Reading Time",
                "RoBERTa-QEye-W": "RoBERTa-Eye-W",
                "RoBERTa-QEye-F": "RoBERTa-Eye-F",
                "FSE": "BEyeLSTM - NT",
                "LRDiane": "Log. Regression",
                "PLMAS": "PLM-AS",
                "PostFusion": "PostFusion-Eye",
                "MAG": "MAG-Eye",
                "Dummy": "Majority Class",
            }
            # Plot the mean ROC curve
            
            line_color = 'black' if model_name_mapping.get(model_name, model_name) == "Majority Class" else None
            # # choose a different color for "Reading Time"
            # line_color = 'darkgreen' if model_name_mapping.get(model_name, model_name) == "Reading Time" else line_color
            line_color = 'dodgerblue' if model_name_mapping.get(model_name, model_name) == "RoBERTa-Eye-F" else line_color
            ax.plot(
                mean_fpr,
                mean_tpr,
                lw=2,
                alpha=0.8,
                label=f"{model_name_mapping.get(model_name, model_name)} ({mean_auc:.2f} ± {std_auc:.2f})",
                color=line_color,
            )

            # Plot the standard deviation as a shaded area
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                alpha=0.1,
            )

        # Plot the random guess line
        # ax.plot([0, 1], [0, 1], linestyle="--", color="black")

        # Add labels and title
        ax.set_xlabel("False Positive Rate", fontsize=14)
        # if i == 0:
        ax.set_ylabel("True Positive Rate", fontsize=14)

        regime_map = {
            "new_item": "New Item",
            "new_subject": "New Participant",
            "new_item_and_subject": "New Item and participant",
        }
        ax.set_title(f"{regime_map[regime]}", fontsize=14)

        # Add legend for each subplot
        ax.legend(loc="lower right")
    
    # increase tick font size
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        
    # plt.suptitle("ROC Curves by Model and Evaluation Regime")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    print(
        f"Saving figure to {base_path / 'ROC_Curves_by_Model_and_Evaluation_Regime.pdf'}"
    )
    plt.savefig(base_path / "ROC_Curves_by_Model_and_Evaluation_Regime.pdf")
    plt.show()


def confusion_matrix_by_regime(
    res, unique_eval_regimes=["new_item", "new_subject", "new_item_and_subject"]
) -> None:
    # Create a figure with three subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    fig.suptitle("Confusion Matrices by Evaluation Regime")
    for i, regime in enumerate(unique_eval_regimes):
        # Filter the DataFrame for the current eval_regime
        regime_res = res[res["eval_regime"] == regime]

        # Compute the confusion matrix
        cm = confusion_matrix(
            regime_res["label"], regime_res["binary_prediction"], normalize="true"
        )

        # Create a heatmap using seaborn in the corresponding subplot
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=["Gathering", "Hunting"],
            yticklabels=["Gathering", "Hunting"],
            ax=axes[i],
        )
        axes[i].set_title(f"{regime}")
        axes[i].set_ylabel("Actual Label")
        axes[i].set_xlabel("Predicted Label")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def classification_report_by_regime(
    res, unique_eval_regimes=["new_item", "new_subject", "new_item_and_subject"]
) -> None:
    for regime in unique_eval_regimes:
        # Filter the DataFrame for the current eval_regime
        regime_res = res[res["eval_regime"] == regime]
        print(f"Classification_report {regime}")
        print(
            classification_report(
                regime_res["label"],
                regime_res["binary_prediction"],
                target_names=["Gathering", "Hunting"],
            )
        )


def fold_level_roc_curve(
    res, unique_eval_regimes=["new_item", "new_subject", "new_item_and_subject"]
) -> None:
    # Assuming 'res' is a DataFrame that contains the 'fold_index' column
    unique_folds = res["fold_index"].unique()
    for i, regime in enumerate(unique_eval_regimes):
        # Filter the DataFrame for the current eval_regime
        regime_res = res[res["eval_regime"] == regime]
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        plt.figure(figsize=(8, 8))

        for fold in unique_folds:
            # Filter the DataFrame for the current fold
            fold_res = regime_res[regime_res["fold_index"] == fold]

            # Compute the false positive rate, true positive rate, and thresholds
            fpr, tpr, _ = roc_curve(fold_res["label"], fold_res["prediction_prob"])

            # Compute the AUROC score
            auroc = roc_auc_score(fold_res["label"], fold_res["prediction_prob"])
            aucs.append(auroc)

            # Interpolate the TPRs to the mean FPRs
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

            # Plot the AUROC curve for the current fold
            plt.plot(
                fpr, tpr, lw=1, alpha=0.3, label=f"Fold {fold} (AUROC = {auroc:.2f})"
            )

        # Compute mean and standard deviation of TPRs and AUROC values
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        # Plot the mean ROC curve
        plt.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        # Plot the standard deviation as a shaded area
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        # Plot the random guess line
        plt.plot([0, 1], [0, 1], linestyle="--", color="r")

        # Add labels and title
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"AUROC Curves for {regime}")

        # Add legend
        plt.legend(loc="lower right")
        plt.show()


def prepare_data(model_df, params, re_cols, outcome):
    model_df_input = model_df[params + re_cols + [outcome]].copy()
    return model_df_input


def standardize_features(model_df_input, params, with_std):
    scaler = StandardScaler(with_std=with_std)
    model_df_input[params] = scaler.fit_transform(model_df_input[params])
    print(f"Standardized coefficients with_std={with_std}")
    return model_df_input


def construct_formula(outcome, params, random_effects_structure):
    concatenated_params = "+".join(params)

    if random_effects_structure == "full":
        formula = f"{outcome} ~ {concatenated_params} + (1 + {concatenated_params} | subject_id) + (1 + {concatenated_params} | unique_paragraph_id) + (1 + {concatenated_params} | eval_regime)"
    elif random_effects_structure == "nested":
        random_effects_subject = " + ".join(
            [f"(1 + {param} | subject_id)" for param in params]
        )
        random_effects_paragraph = " + ".join(
            [f"(1 + {param} | unique_paragraph_id)" for param in params]
        )
        random_effects_eval_regime = " + ".join(
            [f"(1 + {param} | eval_regime)" for param in params]
        )
        formula = f"{outcome} ~ {concatenated_params} + {random_effects_subject} + {random_effects_paragraph} + {random_effects_eval_regime}"
    elif random_effects_structure == "crossed":
        random_effects_subject = " + ".join(
            [f"(1 | subject_id) + ({param} | subject_id)" for param in params]
        )
        random_effects_paragraph = " + ".join(
            [
                f"(1 | unique_paragraph_id) + ({param} | unique_paragraph_id)"
                for param in params
            ]
        )
        random_effects_eval_regime = " + ".join(
            [f"(1 | eval_regime) + ({param} | eval_regime)" for param in params]
        )
        formula = f"{outcome} ~ {concatenated_params} + {random_effects_subject} + {random_effects_paragraph} + {random_effects_eval_regime}"
    elif random_effects_structure == "intercept":
        formula = f"{outcome} ~ {concatenated_params} + (1 | subject_id) + (1 | unique_paragraph_id) + (1 | eval_regime)"
    else:
        raise ValueError("Invalid random effects structure")
    print(f"Random effects structure: {random_effects_structure}")
    print(f"Formula: {formula}")
    return formula


def get_outcome_variable(is_correct_05):
    if is_correct_05:
        outcome = "is_correct_05"
        print("Using is_correct_05 as outcome.")
    else:
        outcome = "is_correct"
        print("Using is_correct as outcome.")
    return outcome


def remove_nan_values(model_df_input):
    if model_df_input.isnull().values.any():
        nan_cols = model_df_input.columns[model_df_input.isnull().any()].tolist()
        total_rows_before = model_df_input.shape[0]
        model_df_input = model_df_input.dropna()
        total_rows_after = model_df_input.shape[0]
        rows_removed = total_rows_before - total_rows_after
        print(f"Dropped NaN values coming from columns: {nan_cols}")
        print(f"Removed {rows_removed} rows out of {total_rows_before} total rows.")
    return model_df_input


def map_pvalue_to_asterisks(pvalue: float) -> str:
    if pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    else:
        return "n.s."
