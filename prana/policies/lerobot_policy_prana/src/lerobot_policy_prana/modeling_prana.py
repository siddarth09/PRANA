
"""
PRANA Act0 Policy

This is a 1:1 ACT-compatible policy. We reuse the ACT architecture (including VAE)

"""


from collections import deque
from dataclasses import dataclass
from itertools import chain 
from typing import Any,Callable 

import torch 
import torch.nn.functional as F 
import torchvision
import einops 
import math 
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
    config_class = PranaAct0Config #Tells the lerobot during training which config class to associate with
    name = "prana" # The name its gets registered as 

    def __init__(self,config:PranaAct0Config):

        super().__init__(config)
        config.validate_features() #Validates all the states that's required by ACT policy 
        self.config = config 

        self.model = PranaAct0Net(config)
        # Action queue basically 
        if config.temporal_ensemble_coeff is not None: 
            self.temporal_ensembler = PranaTemporalEnsembler(config.temporal_ensemble_coeff,config.chunk_size)

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

        l1_loss = (F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)).mean()
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
    



class PranaTemporalEnsembler:
        """
        Online exponential averaging over overlapping action chunks.
        Each update consumes and returns the next action to execute.

        Each timestep actions:(B,chunk_size,action_dim) is called 
        Instead of storing all past chunks, we maintain current average sequence with number of 
        predictions contributed per timestep

        Return only the first action, shift the rest to ensemble 
        """
        def __init__(self,temporal_ensembler_coeff:float,chunk_size:int)->None:

            self.chunk_size = chunk_size

            self.ensemble_weights = torch.exp(-temporal_ensembler_coeff*torch.arange(chunk_size))

            self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights,dim=0)

            self.reset()

        def reset(self):
            """
            Reseting the internal state
            """
            self.ensembled_actions: Tensor | None = None 

            self.ensembled_actions_count:Tensor |None = None 


        def update(self,actions:Tensor)->Tensor:

            """
            Updating temporal actions with a predicted chunk 
            """
            device = actions.device 
            self.ensemble_weights = self.ensemble_weights.to(device)
            self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device)

            if self.ensembled_actions is None:
                # Only for first timestep, to initialize with the first prediction
                self.ensembled_actions = actions.clone()
                self.ensembled_actions_count = torch.ones((self.chunk_size,), dtype=torch.long, device=device
)
            else: 
                # Online update for overlapping actions 

                self.ensembled_actions *=self.ensemble_weights_cumsum[self.ensembled_actions_count-1]
                self.ensembled_actions += (
                        actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
                    )
                self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
                

                # Incrementing the counts

                self.ensembled_actions_count = torch.clamp(
                    self.ensembled_actions_count+1,
                    max = self.chunk_size,
                )


                # Appending the new action chunk 
                self.ensembled_actions = torch.cat([self.ensembled_actions,actions[:,-1:]],
                                                  dim =1,)
                
                self.ensembled_actions_count = torch.cat([
                    self.ensembled_actions_count,torch.ones_like(self.ensembled_actions_count[-1:]),

                ],
                dim = 0,)


            action = self.ensembled_actions[:,0]
            self.ensembled_actions = self.ensembled_actions[:,1:]
            self.ensembled_actions_count = self.ensembled_actions_count[1:]


            return action 
        

def _cfg_get(config: PranaAct0Config, name: str, *aliases: str):
    """Fetch config attr with fallbacks for naming mismatches."""
    if hasattr(config, name):
        return getattr(config, name)
    for a in aliases:
        if hasattr(config, a):
            return getattr(config, a)
    raise AttributeError(f"Config is missing '{name}' (and aliases {aliases}).")

# ======================================
# The Heart 
# ======================================

class PranaAct0Net(nn.Module):
    """
    PranaAct0Net = Implementation of Action chunking Transformer 

    Token ordering into main transformer encoder:
    [latent, (robot_state), (env_state), (image_feature_pixels...)]
    Decoder uses DETR-style learned queries for chunk_size steps.
    """

    def __init__(self,config:PranaAct0Config):
        super().__init__()
        self.config = config 
        dim_model = _cfg_get(config,"dim_model")
        chunk_size = _cfg_get(config, "chunk_size")
        latent_dim = _cfg_get(config, "latent_dim")
        dropout = _cfg_get(config, "dropout")


        if config.use_vae:
            self.vae_encoder = PranaEncoder(config,is_vae_encoder = True)
            self.vae_encoder_cls_embed = nn.Embedding(1,dim_model)

            if getattr(config,"robot_state_feature",None):
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    config.robot_state_feature.shape[0],dim_model
                )

            self.vae_encoder_action_input_proj = nn.Linear(
                config.action_feature.shape[0],
                dim_model,
            )

            self.vae_encoder_latent_output_proj = nn.Linear(dim_model,latent_dim*2)


            # Sinusoidal pos embed for [cls,(robot_state),*action_seq]
            n_tokens = 1 + chunk_size
            if getattr(config,"robot_state_feature",None):
                n_tokens +=1 

            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(n_tokens,dim_model).unsqueeze(0)
            )

        
        if config.image_features:
            backbone_model = getattr(torchvision.models,config.vision_backbone)(
                replace_stride_with_dilation = [False,False,config.replace_final_stride_with_dilation],
                weights = config.pretrained_backbone_weights,
                norm_layer= FrozenBatchNorm2d
            )

            self.backbone = IntermediateLayerGetter(backbone_model,return_layers={"layer4":"feature_map"})
        else:
            backbone_model= None 
            self.backbone = None 


        # ===================
        # core
        # ==================


        self.encoder = PranaEncoder(config,is_vae_encoder = False)
        self.decoder = PranaDecoder(config)

        if getattr(config, "robot_state_feature", None):
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], dim_model
            )
        if getattr(config, "env_state_feature", None):
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_feature.shape[0], dim_model
            )

        self.encoder_latent_input_proj = nn.Linear(latent_dim, dim_model)

        if config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features,dim_model,kernel_size = 1)

        n_1d_tokens = 1
        if getattr(config,"robot_state_feature",None):
            n_1d_tokens += 1 
        if getattr(config,"env_state_feature",None):
            n_1d_tokens +=1 
        
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens,dim_model)

        if config.image_features:
            self.encoder_cam_feat_pos_embed = PranaSinusoidalPositionEmbedding2d(dim_model//2)

        
        self.decoder_pos_embed = nn.Embedding(chunk_size,dim_model)
        self.action_head = nn.Linear(dim_model,config.action_feature.shape[0])

        self._reset_parameters()

    
    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    
    def forward(self,batch: dict[str,Tensor])->tuple[Tensor,tuple[Tensor, Tensor] | tuple[None, None]]:
        """
        Returns:
            Actions: (B,chunk_size,action_dim)
            (mu,log_sigma_x2): (B,latent_dim) each or (None,None) if not using VAE
        """

        if self.config.use_vae and self.training:
            assert ACTION in batch, "actions must be provided when training "

        # Batch Size determination 

        if OBS_IMAGES in batch:

            batch_size = batch[OBS_IMAGES][0].shape[0]
            device = batch[OBS_IMAGES][0].device 
        else:

            batch_size = batch[OBS_ENV_STATE].shape[0]
            device = batch[OBS_ENV_STATE].device 

        chunk_size = _cfg_get(self.config, "chunk_size")
        latent_dim = _cfg_get(self.config, "latent_dim")
        dim_model = _cfg_get(self.config, "dim_model")

        # VAE latent preparation 

        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight,"1 d -> b 1 d",b=batch_size)
            if getattr(self.config, "robot_state_feature", None):
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B,S,D)

            if getattr(self.config, "robot_state_feature", None):
                vae_in = torch.cat([cls_embed, robot_state_embed, action_embed], dim=1)
            else:
                vae_in = torch.cat([cls_embed, action_embed], dim=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            n_prefix = 2 if getattr(self.config, "robot_state_feature", None) else 1
            prefix_is_pad = torch.full((batch_size, n_prefix), False, device=device)

            
            action_is_pad = batch.get("action_is_pad", None)
            if action_is_pad is None:
                raise KeyError("Batch must contain 'action_is_pad' when training ACT/PRANA with VAE.")

            key_padding_mask = torch.cat([prefix_is_pad, action_is_pad], dim=1)

            cls_token_out = self.vae_encoder(
                vae_in.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # (B,D)

            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, :latent_dim]
            log_sigma_x2 = latent_pdf_params[:, latent_dim:]  # matches ACT: 2*log(sigma)

            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros((batch_size, latent_dim), dtype=torch.float32, device=device)
        
        encoder_tokens = []
        encoder_pos = []

        # ---- latent token ----
        latent_token = self.encoder_latent_input_proj(latent_sample)  # (B, D)
        encoder_tokens.append(latent_token)
        encoder_pos.append(self.encoder_1d_feature_pos_embed.weight[0].unsqueeze(0))  # (1, D)

        pos_idx = 1

        # ---- robot state token ----
        if getattr(self.config, "robot_state_feature", None):
            token = self.encoder_robot_state_input_proj(batch[OBS_STATE])
            encoder_tokens.append(token)
            encoder_pos.append(self.encoder_1d_feature_pos_embed.weight[pos_idx].unsqueeze(0))
            pos_idx += 1

        # ---- env state token ----
        if getattr(self.config, "env_state_feature", None):
            token = self.encoder_env_state_input_proj(batch[OBS_ENV_STATE])
            encoder_tokens.append(token)
            encoder_pos.append(self.encoder_1d_feature_pos_embed.weight[pos_idx].unsqueeze(0))
            pos_idx += 1

        # ---- image tokens ----
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                raw = self.backbone(img)["feature_map"]
                cam_pos = self.encoder_cam_feat_pos_embed(raw).to(dtype=raw.dtype)
                cam_features = self.encoder_img_feat_input_proj(raw)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos = einops.rearrange(cam_pos, "b c h w -> (h w) b c")

                encoder_tokens.extend(list(cam_features))
                encoder_pos.extend(list(cam_pos))

        
        encoder_tokens = torch.stack(encoder_tokens, dim=0)
        encoder_pos = torch.stack(encoder_pos, dim=0)

        # -------------------------
        # Transformer forward
        # -------------------------
        encoder_out = self.encoder(encoder_tokens, pos_embed=encoder_pos)

        decoder_in = torch.zeros(
            (chunk_size, batch_size, dim_model),
            dtype=encoder_pos.dtype,
            device=encoder_pos.device,
        )

        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_pos,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # (B, S, D) then action head -> (B, S, A)
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)
    

class PranaEncoder(nn.Module):
    """Encoder layer"""

    def __init__(self,config:PranaAct0Config,is_vae_encoder:bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder

        n_layers = _cfg_get(
            config,
            "n_vae_encoder_layers" if is_vae_encoder else "n_encoder_layers",
            "n_vae_encoder_layers" if is_vae_encoder else "n_encoder_layer",
        )

        self.layers = nn.ModuleList([PranaEncoderLayer(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(_cfg_get(config,"dim_model")) if config.pre_norm else nn.Identity()



    def forward(self,x:Tensor,pos_embed:Tensor | None = None, key_padding_mask: Tensor | None = None)-> Tensor:
        for layer in self.layers:
            x = layer(x,pos_embed=pos_embed,key_padding_mask = key_padding_mask)
        x = self.norm(x)

        return x 
    



class PranaEncoderLayer(nn.Module):
    def __init__(self, config: PranaAct0Config):
        super().__init__()
        dim_model = _cfg_get(config, "dim_model")
        n_heads = _cfg_get(config, "n_heads")
        dropout = _cfg_get(config, "dropout")
        dim_ff = _cfg_get(config, "dim_feedforward")
        act_name = _cfg_get(config, "feedforward_activation", "activation", "ff_activation")

        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)

        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(act_name)
        self.pre_norm = _cfg_get(config, "pre_norm")

    def forward(self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)

        if not self.pre_norm:
            x = self.norm2(x)
        return x


class PranaDecoder(nn.Module):
    """Runs multiple decoder layers then normalization."""

    def __init__(self, config: PranaAct0Config):
        super().__init__()
        n_layers = _cfg_get(config, "n_decoder_layers", "n_decoder_layer")
        self.layers = nn.ModuleList([PranaDecoderLayer(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(_cfg_get(config, "dim_model"))

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        x = self.norm(x)
        return x
    

class PranaDecoderLayer(nn.Module):
    def __init__(self, config: PranaAct0Config):
        super().__init__()
        dim_model = _cfg_get(config, "dim_model")
        n_heads = _cfg_get(config, "n_heads")
        dropout = _cfg_get(config, "dropout")
        dim_ff = _cfg_get(config, "dim_feedforward")
        act_name = _cfg_get(config, "feedforward_activation", "activation", "ff_activation")

        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)

        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = get_activation_fn(act_name)
        self.pre_norm = _cfg_get(config, "pre_norm")

    @staticmethod
    def _maybe_add_pos(x: Tensor, pos: Tensor | None) -> Tensor:
        return x if pos is None else x + pos

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)

        q = k = self._maybe_add_pos(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.cross_attn(
            query=self._maybe_add_pos(x, decoder_pos_embed),
            key=self._maybe_add_pos(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)

        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)

        if not self.pre_norm:
            x = self.norm3(x)
        return x
    
def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings (Attention is All You Need)."""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class PranaSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings (same math as ACT)."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,C,H,W)
        not_mask = torch.ones_like(x[0, :1])  # (1,H,W)
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inv_freq = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inv_freq
        y_range = y_range.unsqueeze(-1) / inv_freq

        pos_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (1,C,H,W)
        return pos
    
def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

