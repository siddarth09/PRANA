import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor
from collections import deque

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from prana.model.modeling import PranaVLA
from prana.model.configuration_prana import PranaConfig

class PranaPolicy(PreTrainedPolicy):
    config_class = PranaConfig
    name = "prana_v1"

    def __init__(self, config: PranaConfig):
        super().__init__(config)
        self.config = config

        state_dim = config.input_features[OBS_STATE].shape[0] if OBS_STATE in config.input_features else 6
        action_dim = config.output_features[ACTION].shape[0] if ACTION in config.output_features else 6
        
        self.model = PranaVLA(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=config.chunk_size,
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size
        )
        self.reset()

    def get_optim_params(self) -> list[dict]:
        return [{"params": self.model.parameters(), "lr": self.config.optimizer_lr}]

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # --- THE FIX: Resize the raw 480x640 frames to 224x224 ---
        img_table = TF.resize(batch["observation.images.table"], [224, 224], antialias=True)
        img_wrist = TF.resize(batch["observation.images.wrist"], [224, 224], antialias=True)
        images = [img_table, img_wrist]
        
        states = batch.get(OBS_STATE, torch.zeros(images[0].shape[0], 6, device=images[0].device))
        target_actions = batch[ACTION]
        tokens = torch.zeros((images[0].shape[0], 16), dtype=torch.long, device=images[0].device)

        predicted_actions = self.model(images, states, tokens)
        
        l1_loss = F.l1_loss(predicted_actions, target_actions, reduction="none")
        if "action_is_pad" in batch:
            l1_loss = (l1_loss * ~batch["action_is_pad"].unsqueeze(-1)).mean()
        else:
            l1_loss = l1_loss.mean()

        return l1_loss, {"l1_loss": l1_loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        
        # --- THE FIX: Resize during inference too ---
        img_table = TF.resize(batch["observation.images.table"], [224, 224], antialias=True)
        img_wrist = TF.resize(batch["observation.images.wrist"], [224, 224], antialias=True)
        images = [img_table, img_wrist]
        
        states = batch.get(OBS_STATE, torch.zeros(images[0].shape[0], 6, device=images[0].device))
        tokens = torch.zeros((images[0].shape[0], 16), dtype=torch.long, device=images[0].device)

        predicted_chunk = self.model(images, states, tokens)
        return predicted_chunk

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            # Neatly call predict_action_chunk so we don't repeat the resize logic
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
            
        return self._action_queue.popleft()