from dataclasses import dataclass, field

from src.configs.constants import BackboneNames, ModelNames, ConfigName
from src.configs.model_args.base_model_args import BaseModelArgs, BaseModelParams
from src.configs.prediction_modes import ConditionPredCfg, PredCfg
from src.configs.utils import register_config

GROUP = ConfigName.MODEL


@dataclass
class PLMASParams(BaseModelParams):
    model_name: ModelNames = ModelNames.PLMAS_MODEL

    # fixation sequence encoder - lstm
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1


@register_config(group=GROUP)
@dataclass
class PLMASArgs(BaseModelArgs):
    batch_size: int = 128
    accumulate_grad_batches: int = 2
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)
    use_fixation_report: bool = True
    text_dim: int = 768 if backbone == BackboneNames.ROBERTA_BASE else 1024
    max_seq_len: int = 311
    max_eye_len: int = 300
    add_contrastive_loss: bool = False
    freeze: bool = True

    # model parameters
    model_params: PLMASParams = field(
        default_factory=lambda: PLMASParams(
            model_name=ModelNames.PLMAS_MODEL,
            class_weights=None,  # Gets overritten by the trainer if not None
        )
    )

    fixation_features: list[str] = field(
        default_factory=lambda: [  #! Keep the order of the first 3 here - the model assumes their existance in this order
            "CURRENT_FIX_INTEREST_AREA_INDEX",
            "CURRENT_FIX_DURATION",
            "CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE",
        ]
    )
    eye_features: list[str] = field(
        default_factory=lambda: [
            "IA_DWELL_TIME",
        ]
    )

    ia_categorical_features: list[str] = field(  # They are exluded
        default_factory=lambda: [
            "Is_Content_Word",
            "Reduced_POS",
            "Entity",
            "POS",
            "Head_Direction",
            "TRIAL_IA_COUNT",
            "TRIAL_IA_COUNT",
            "IA_REGRESSION_OUT_FULL_COUNT",
            "IA_FIXATION_COUNT",
            "IA_REGRESSION_IN_COUNT",
        ]
    )


@register_config(group=GROUP)
@dataclass
class PLMASCondPredArgs(PLMASArgs):
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)
    batch_size: int = 16
    accumulate_grad_batches: int = 1
    max_seq_len: int = 228  # in use only for creating TextDataset
    max_eye_len: int = 258
