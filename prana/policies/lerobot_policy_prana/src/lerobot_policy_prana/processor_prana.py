from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
)

from .configuration_prana import PranaAct0Config


def make_prana_pre_post_processors(
    config: PranaAct0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Creates the pre- and post-processing pipelines for the PRANA Act0 policy.

    Mirrors ACT exactly:
    - pre: rename → batch → device → normalize
    - post: unnormalize → cpu → transition conversion
    """

    # -------------------------
    # Pre-processing pipeline
    # -------------------------
    input_steps = [
        # PRANA datasets already use canonical names
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={
                **config.input_features,
                **config.output_features,
            },
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]

    # -------------------------
    # Post-processing pipeline
    # -------------------------
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
