from dataclasses import dataclass, field


from src.configs.constants import (
    MLModelNames,
    ConfigName,
    ItemLevelFeaturesModes,
)
from src.configs.model_args.base_model_args_ml import BaseMLModelArgs, BaseMLModelParams
from src.configs.prediction_modes import (
    ConditionPredCfg,
    IsCorrectPredCfg,
    PredCfg,
    QPredCfg,
)
from src.configs.utils import register_config

GROUP = ConfigName.MODEL


@dataclass
class LogisticRegressionMLParams(BaseMLModelParams):
    concat_or_duplicate: str = "concat"

    model_name: MLModelNames = MLModelNames.LOGISTIC_REGRESSION
    sklearn_pipeline: tuple = (
        ("scaler", "sklearn.preprocessing.StandardScaler"),
        ("clf", "sklearn.linear_model.LogisticRegression"),
    )
    # sklearn pipeline params
    #! note the naming convention for the parameters:
    #! sklearn_pipeline_param_<pipline_element_name>__<param_name>

    # clf params
    sklearn_pipeline_param_clf__C: float = 2.0
    sklearn_pipeline_param_clf__fit_intercept: bool = True
    sklearn_pipeline_param_clf__penalty: str = "l2"
    sklearn_pipeline_param_clf__solver: str = "lbfgs"
    sklearn_pipeline_param_clf__random_state: int = 1
    sklearn_pipeline_param_clf__max_iter: int = 1000
    sklearn_pipeline_param_clf__class_weight: str = "balanced"

    # scaler params
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True


@register_config(group=GROUP)
@dataclass
class LogisticRegressionMLArgs(BaseMLModelArgs):
    batch_size: int = -1
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams()
    )
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)
    #! note logistic regression is for binary classification
    text_dim: int = 768  # ?
    max_seq_len: int = 228  # in use only for creating TextDataset
    max_eye_len: int = 258
    use_fixation_report: bool = True
    backbone: str = "roberta-base"


# Condition Prediction
@register_config(group=GROUP)
@dataclass
class LogisticRegressionCondPredMLArgs(LogisticRegressionMLArgs):
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)


@register_config(group=GROUP)
@dataclass
class LogisticRegressionCondPredDianeMLArgs(LogisticRegressionCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_modes=[
                ItemLevelFeaturesModes.DIANE,
            ],
        )
    )


@register_config(group=GROUP)
@dataclass
class LogisticRegressionCondPredAvgDwellTimeMLArgs(LogisticRegressionCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_modes=[
                ItemLevelFeaturesModes.AVERAGE_DWELL_TIME,
            ],
        )
    )


@register_config(group=GROUP)
@dataclass
class LogisticRegressionCondPredReadingSpdMLArgs(LogisticRegressionCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_modes=[
                ItemLevelFeaturesModes.READING_SPEED,
            ],
        )
    )


# IsCorrect Prediction Logistic Regression
@register_config(group=GROUP)
@dataclass
class LogisticRegressionIsCorrectPredMLArgs(LogisticRegressionMLArgs):
    prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)


@register_config(group=GROUP)
@dataclass
class LogisticRegressionIsCorrectPredEMNLPMLArgs(LogisticRegressionIsCorrectPredMLArgs):
    # uses david, lenna, diane features
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_modes=[
                ItemLevelFeaturesModes.LENNA,
                ItemLevelFeaturesModes.DIANE,
            ],
        )
    )


@register_config(group=GROUP)
@dataclass
class LogisticRegressionIsCorrectPredDianeEMNLPMLArgs(
    LogisticRegressionIsCorrectPredMLArgs
):
    # uses david, lenna, diane features
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_modes=[
                ItemLevelFeaturesModes.DIANE,
            ],
        )
    )


# Question Prediction Logistic Regression
@register_config(group=GROUP)
@dataclass
class LogisticRegressionQPredMLArgs(LogisticRegressionMLArgs):
    prediction_config: PredCfg = field(default_factory=QPredCfg)
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            concat_or_duplicate="duplicate",
            use_DWELL_TIME_WEIGHTED_PARAGRAPH_DOT_QUESTION_CLS_NO_CONTEXT=True,
            use_QUESTION_RELEVANCE_SPAN_DOT_IA_DWELL_NO_CONTEXT=True,
            item_level_features_modes=[],
        )
    )
    use_fixation_report: bool = False
    add_beyelstm_features: bool = False

    ia_features: list[str] = field(default_factory=lambda: ["IA_DWELL_TIME"])
    fixation_features: list[str] = field(default_factory=lambda: [])
    ia_categorical_features: list[str] = field(default_factory=lambda: [])
