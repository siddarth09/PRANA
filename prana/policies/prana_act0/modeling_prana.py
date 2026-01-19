
"""
PRANA Act0 Policy

This is a 1:1 ACT-compatible policy. We reuse the ACT architecture (including VAE)

"""


from collections import deque
from dataclasses import dataclass
from itertools import chain 
from typing import Any 

import torch 
import torch.nn.functional as F 
import torchvision
import einops 
import numpy as np 
from torch import nn, Tensor
#IntermediateLayerGetter: Grabs intermediate feature maps from a torchvision backbone
from torchvision.models._utils import IntermediateLayerGetter
#FrozenBatchNorm2d: freeze BN Stats 
from torchvision.ops.misc import FrozenBatchNorm2d 
from lerobot.policies.pretrained import PreTrainedPolicy 
from lerobot.utils.constants import ACTION,OBS_ENV_STATE,OBS_IMAGES,OBS_STATE

from .configuration_prana import PranaAct0Config



class PranaAct0Policy(PreTrainedPolicy):
    config = PranaAct0Config #Tells the lerobot during training which config class to associate with
    name = "prana_act0" # The name its gets registered as 

    def __init__(self,config:PranaAct0Config):

        super().__init__(config)
        config.validate_features() #Validates all the states that's required by ACT policy 
        self.config = config 

        self.model = PranaAct0Config(config)
        # Action queue basically 
        if config.temporal_ensemble_coeff is not None: 
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff)

        self.reset() #Initializes internal inference state 


    
    def get_optim_params(self):

        """
        Builds the optimizer parameter group
        where everything except the vision backbone uses the learning rate
        """
        return [
            {
                "params":[
                    p for n,p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]
    


    def reset(self):
        if self.config.temporal_ensemble_coeff is not None: 
            self.temporal_ensembler.reset()
        else: 
            self._action_queue = deque([],maxlen=self.config.n_action_steps)


    @torch.no_grad()

    def select_action(self,batch:dict[str,Tensor]) -> Tensor:

        # Function used during inference 
        self.eval()

        if self.config.temporal_ensemble_coeff is not None: 
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)
        
        # Predict a chunk, keep only first n_action_steps, store them as a queue of per time stamp actions
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:,: self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0,1))

        return self._action_queue.popleft()
    
    @torch.no_grad()

    def predict_action_chunk(self,batch:dict[str,Tensor])->Tensor:
        self.eval()

        if self.config.image_features:
            batch = dict(batch)

            batch[OBS_IMAGES] = [batch[k] for k in self.config.camera_order]

        actions = self.model(batch)[0]

        return actions 
    

    def forward(self,batch:dict[str,Tensor])-> tuple[Tensor,dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[k] for k in self.config.camera_order]

        actions_hat, (mu_hat,log_sigma_x2_hat) = self.model(batch) 

        l1_loss =( F.l1_loss(batch[ACTION],actions_hat,reduction="None") * ~batch["actions_is_pad"].unsqueeze(-1)).mean()
        loss_dict = {"l1_loss": float(l1_loss.item())}

        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = float(mean_kld.item())
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict
            