from dataclasses import dataclass,field 
from typing import Dict,Optional

from lerobot.configs.policies import PreTrainedConfig 
from lerobot.configs.types import NormalizationMode 
from lerobot.optim.optimizers import AdamWConfig 


@PreTrainedConfig.register_subclass("prana_v0")
@dataclass 

class PranaAct0Config(PreTrainedConfig):
    """
    PranaAct0Config

    This is a direct implementation of ACT0 with the help of lerobot libraries 
    """

    # ---------------------------
    # Input / output structure.
    # ---------------------------

    n_obs_steps:int = 1 #How many observation steps the policy expects 
    chunk_size:int = 100 # How many actions the model predicts in one forward pass 
    n_action_steps:int = 100  #How many actions you actually  execute from that predicted chunk before requerying

    #Normalizing all the features 
    normalization_mapping: dict[str,NormalizationMode] = field(
        default_factory=lambda:{
            "VISUAL":NormalizationMode.MEAN_STD,
            "STATE":NormalizationMode.MEAN_STD,
            "ACTION":NormalizationMode.MEAN_STD
    
        }
    )

    # ---------------------------
    # Architecture.
    # ---------------------------

    # Vision Backbone 

    vision_backbone:str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1k_V1"
    replace_final_stride_with_dilation:int = False 

    # Transform Layers 
    pre_norm: bool = False #Applying layernorm 
    dim_model: int = 512  #token embedding dimension
    n_heads: int = 8  #number of attention heads 
    dim_feedforward:int = 3200 # hidden size inside the mlp part of transformer blocks
    feedforward_activation= str= "relu" #Mlp nonlinearity 
    n_encoder_layer: int = 4  #How many encoder layers in the main act transformer 

    n_decoder_layer: int = 1 #How many decoder layers in the transformer 


    #Vision Action encoder 
    use_vae : bool = True  
    latent_dim : int = 32 
    n_vae_encoder_layers: int = 4 

    temporal_ensemble_coeff: float | None = None 

    # Training and loss computation 

    dropout : float = 0.1 
    kl_weight: float = 10.0 
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    camera_order: list[str] = field(
        default_factory=lambda: ["observation.images.table", "observation.images.wrist"]
    )



    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )


    def get_optimizer_preset(self)->AdamWConfig:
        # Building the optimizer
        return AdamWConfig(lr=self.optimizer_lr,
                           weight_decay=self.optimizer_weight_decay,
        )
    
    def get_scheduler_preset(self):
        # No lr scheduler 
        return None 
    

    def validate_features(self):
        if not self.image_features and not self.env_state_feature:
            raise ValueError("Must provide atleast one image")
        
        
    @property 
    def observation_delta_indices(self) -> None:
        return None 
    
    @property 
    def action_delta_indices(self)->list:
        return list(range(self.chunk_size))
    
    @property 
    def reward_delta_indices(self)->None:
        return None 
    