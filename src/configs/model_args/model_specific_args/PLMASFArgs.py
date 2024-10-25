from dataclasses import dataclass, field

from src.configs.constants import (
    BackboneNames,
    ItemLevelFeaturesModes,
    ModelNames,
    ConfigName,
)
from src.configs.model_args.base_model_args import BaseModelArgs, BaseModelParams
from src.configs.prediction_modes import ConditionPredCfg, PredCfg
from src.configs.utils import register_config

GROUP = ConfigName.MODEL


@dataclass
class PLMASfParams(BaseModelParams):
    model_name: ModelNames = ModelNames.PLMASF_MODEL

    lstm_hidden_size: int = 128

    # fixation sequence encoder - lstm
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1


@register_config(group=GROUP)
@dataclass
class PLMASfArgs(BaseModelArgs):
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
    model_params: PLMASfParams = field(
        default_factory=lambda: PLMASfParams(
            model_name=ModelNames.PLMASF_MODEL,
            class_weights=None,  # Gets overritten by the trainer
        )
    )

    fixation_features: list[str] = field(
        default_factory=lambda: [  #! Keep the order
            "CURRENT_FIX_INTEREST_AREA_INDEX",
            "CURRENT_FIX_DURATION",
            "CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE",
            "CURRENT_FIX_Y",
            "NEXT_SAC_DURATION",
            "NEXT_SAC_END_X",
            "NEXT_SAC_START_X",
            "NEXT_SAC_END_Y",
            "NEXT_SAC_START_Y",
        ]
    )
    eye_features: list[str] = field(
        default_factory=lambda: [  #! Keep the order
            "IA_DWELL_TIME",
            "IA_FIRST_RUN_LANDING_POSITION",
            "IA_LAST_RUN_LANDING_POSITION",
            "IA_FIRST_FIXATION_DURATION",
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

    def __post_init__(self):
        super().__post_init__()


@register_config(group=GROUP)
@dataclass
class PLMASfCondPredArgs(PLMASfArgs):
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)
    batch_size: int = 16
    accumulate_grad_batches: int = 1
    max_seq_len: int = 228  # in use only for creating TextDataset
    max_eye_len: int = 258


@register_config(group=GROUP)
@dataclass
class HallerWFFCondPredArgs(PLMASfCondPredArgs):
    # model parameters
    model_params: PLMASfParams = field(
        default_factory=lambda: PLMASfParams(
            model_name=ModelNames.HALLER_W_FF_MODEL,
            item_level_features_modes=[ItemLevelFeaturesModes.DAVID],
            class_weights=None,  # Gets overritten by the trainer
        )
    )
    add_beyelstm_features: bool = True
    fixation_features: list[str] = field(
        default_factory=lambda: [
            "CURRENT_FIX_DURATION",
            "CURRENT_FIX_PUPIL",
            "CURRENT_FIX_X",
            "CURRENT_FIX_Y",
            "NEXT_FIX_INTEREST_AREA_INDEX",
            "CURRENT_FIX_INTEREST_AREA_INDEX",
        ]
    )
    eye_features: list[str] = field(
        default_factory=lambda: [
            "TRIAL_IA_COUNT",
            "IA_REGRESSION_OUT_FULL_COUNT",
            "IA_FIXATION_COUNT",
            "IA_REGRESSION_IN_COUNT",
            "IA_FIRST_FIXATION_DURATION",
            "IA_DWELL_TIME",
        ]
    )

    word_features: list[str] = field(
        default_factory=lambda: [
            "Is_Content_Word",
            "Reduced_POS",
            "n_Lefts",
            "n_Rights",
            "Distance2Head",
            "Head_Direction",
            "gpt2_Surprisal",
            "Wordfreq_Frequency",
            "Length",
            "Entity",
            "POS",
        ]
    )

    ia_categorical_features: list[str] = field(
        default_factory=lambda: [
            "Is_Content_Word",
            "Reduced_POS",
            "Entity",
            "POS",
            "Head_Direction",
            "TRIAL_IA_COUNT",
            "IA_REGRESSION_OUT_FULL_COUNT",
            "IA_FIXATION_COUNT",
            "IA_REGRESSION_IN_COUNT",
        ]
    )
