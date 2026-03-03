from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig

@PreTrainedConfig.register_subclass("prana_v1")
@dataclass
class PranaConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50  
    
    hidden_dim: int = 256
    vocab_size: int = 256000
    
    # Let LeRobot handle 'image_features' internally, only parse 'camera_order'
    camera_order: list[str] = field(default_factory=lambda: ["observation.images.table", "observation.images.wrist"])
    
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot be greater than chunk_size.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        pass 

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None