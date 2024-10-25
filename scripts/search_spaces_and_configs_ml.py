from dataclasses import dataclass

from scripts.run_wrapper import (
    DataOptions,
    DataPathOptions,
    MLModelOptions,
    MLTrainerOptions,
)

from src.configs.constants import MLModelNames


search_space_by_model_name: dict[MLModelNames, dict] = {
    MLModelNames.LOGISTIC_REGRESSION: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        # pipeline params
                        "sklearn_pipeline_param_clf__C": {
                            "values": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
                        },
                        "sklearn_pipeline_param_clf__fit_intercept": {"values": [True]},
                        "sklearn_pipeline_param_clf__penalty": {"values": ["l2", None]},
                        "sklearn_pipeline_param_clf__solver": {"values": ["lbfgs"]},
                        "sklearn_pipeline_param_clf__random_state": {"values": [1]},
                        "sklearn_pipeline_param_clf__max_iter": {"values": [1000]},
                        "sklearn_pipeline_param_clf__class_weight": {"values": [None]},
                        # scaler params
                        "sklearn_pipeline_param_scaler__with_mean": {"values": [True]},
                        "sklearn_pipeline_param_scaler__with_std": {"values": [True]},
                    },
                },
            },
        }
    },
    MLModelNames.KNN: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        # pipeline params
                        "sklearn_pipeline_param_clf__n_neighbors": {
                            "values": [1, 3, 5, 10, 15, 20, 25, 30]
                        },
                        "sklearn_pipeline_param_clf__weights": {
                            "values": ["uniform", "distance"]
                        },
                        "sklearn_pipeline_param_clf__algorithm": {"values": ["auto"]},
                        "sklearn_pipeline_param_clf__leaf_size": {"values": [30]},
                        "sklearn_pipeline_param_clf__p": {"values": [1, 2, 3, 4, 5, 6]},
                        "sklearn_pipeline_param_clf__metric": {"values": ["minkowski"]},
                        # scaler params
                        "sklearn_pipeline_param_scaler__with_mean": {"values": [True]},
                        "sklearn_pipeline_param_scaler__with_std": {"values": [True]},
                    },
                },
            },
        }
    },
    MLModelNames.DUMMY_CLASSIFIER: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        # pipeline params
                        "sklearn_pipeline_param_clf__strategy": {
                            "values": [
                                "stratified",
                                "most_frequent",
                                "prior",
                                "uniform",
                            ]
                        },
                    },
                },
            },
        }
    },
    MLModelNames.SVM: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        # pipeline params
                        "sklearn_pipeline_param_clf__C": {
                            "values": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
                        },
                        "sklearn_pipeline_param_clf__kernel": {
                            "values": ["rbf", "linear"]
                        },
                        "sklearn_pipeline_param_clf__degree": {"values": [3]},
                        "sklearn_pipeline_param_clf__gamma": {
                            "values": ["scale", "auto", 0.1, 0.01, 0.001, 0.0001]
                        },
                        "sklearn_pipeline_param_clf__coef0": {"values": [0.0]},
                        "sklearn_pipeline_param_clf__shrinking": {"values": [True]},
                        "sklearn_pipeline_param_clf__probability": {"values": [False]},
                        "sklearn_pipeline_param_clf__tol": {"values": [0.001]},
                        "sklearn_pipeline_param_clf__class_weight": {
                            "values": ["balanced", None]
                        },
                        # scaler params
                        "sklearn_pipeline_param_scaler__with_mean": {"values": [True]},
                        "sklearn_pipeline_param_scaler__with_std": {"values": [True]},
                    },
                },
            },
        },
    },
}


@dataclass
class MLRunConfig:
    model_name: MLModelNames
    model_variant: MLModelOptions
    data_variant: DataOptions
    trainer_variant: MLTrainerOptions = "CfirMLVanila"
    data_path: DataPathOptions = "august06"


@dataclass
class HuntingMLRunConfig(MLRunConfig):
    data_variant: DataOptions = "Hunting"


@dataclass
class GatheringMLRunConfig(MLRunConfig):
    data_variant: DataOptions = "Gathering"


@dataclass
class ConditionPredictionMLRunConfig(MLRunConfig):
    data_variant: DataOptions = "NoReread"


is_correct_run_configs: dict[str, MLRunConfig] = {
    # Logistic Regression
    "logreg_emnlp_h_is_corr": HuntingMLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionIsCorrectPredEMNLPMLArgs",
        trainer_variant="CfirMLVanila",
    ),
    "logreg_emnlp_g_is_corr": GatheringMLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionIsCorrectPredEMNLPMLArgs",
        trainer_variant="CfirMLVanila",
    ),
    # Logistic Regression #! Diane features only
    "logreg_diane_emnlp_h_is_corr": HuntingMLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionIsCorrectPredDianeEMNLPMLArgs",
        trainer_variant="CfirMLVanila",
    ),
    "logreg_diane_emnlp_g_is_corr": GatheringMLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionIsCorrectPredDianeEMNLPMLArgs",
        trainer_variant="CfirMLVanila",
    ),
    # KNN
    "knn_emnlp_h_is_corr": HuntingMLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsIsCorrectPredEMNLPMLArgs",
        trainer_variant="CfirMLVanila",
    ),
    "knn_emnlp_g_is_corr": GatheringMLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsIsCorrectPredEMNLPMLArgs",
        trainer_variant="CfirMLVanila",
    ),
    # SVM
    "svm_emnlp_h_is_corr": HuntingMLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineIsCorrectPredEMNLPMLArgs",
        trainer_variant="CfirMLVanila",
    ),
    "svm_emnlp_g_is_corr": GatheringMLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineIsCorrectPredEMNLPMLArgs",
        trainer_variant="CfirMLVanila",
    ),
    # Dummy Classifier
    "dummy_h_is_corr": HuntingMLRunConfig(
        model_name=MLModelNames.DUMMY_CLASSIFIER,
        model_variant="DummyClassifierIsCorrectPredMLArgs",
        trainer_variant="CfirMLVanila",
    ),
    "dummy_g_is_corr": GatheringMLRunConfig(
        model_name=MLModelNames.DUMMY_CLASSIFIER,
        model_variant="DummyClassifierIsCorrectPredMLArgs",
        trainer_variant="CfirMLVanila",
    ),
}

condition_prediction_run_configs: dict[str, MLRunConfig] = {
    "dummy_cond_pred": ConditionPredictionMLRunConfig(
        model_name=MLModelNames.DUMMY_CLASSIFIER,
        model_variant="DummyClassifierCondPredMLArgs",
        trainer_variant="CfirMLVanila",
    ),
    "logreg_cond_pred_diane": ConditionPredictionMLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionCondPredDianeMLArgs",
        trainer_variant="CfirMLVanila",
    ),
    "logreg_cond_pred_AvgDwellTime": ConditionPredictionMLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionCondPredAvgDwellTimeMLArgs",
        trainer_variant="CfirMLVanila",
    ),
    "logreg_cond_pred_ReadingSpeed": ConditionPredictionMLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionCondPredReadingSpdMLArgs",
        trainer_variant="CfirMLVanila",
    ),
}

run_configs: dict[str, MLRunConfig] = {
    **is_correct_run_configs,
    **condition_prediction_run_configs,
}
