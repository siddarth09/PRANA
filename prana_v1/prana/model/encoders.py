import torch 
import torch.nn as nn 
import timm 

class VisionEncoder(nn.Module):
    """
    Process images using lightweight VisionTransformer
    """

    def __init__(self,model_name: str = "vit_tiny_patch16_224",hidden_dim: int = 256):
        # Loading the ViT 
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained = True,
            num_classes = 0,
            global_pool = ''
        )


        # Getting the native hidden size of the chosen ViT
        vit_embed_dim = self.backbone.embed_dim 
        self.proj = nn.Linear(vit_embed_dim,hidden_dim)


    def forward(self, image: torch.Tensor) -> torch.tensor:

        # image shape: [Batch, 3, 224, 224]
        features = self.backbone(image) 
        # features shape: [Batch, 197, vit_embed_dim]

        tokens = self.proj(features)

        return tokens 
    

    



class LanguageEncoder(nn.Module):

    def __init__(self,vocab_size: int = 256000,hidden_dim: int = 256):
        super().__init__()
        # 256000 matches with PaliGemma tokenizer vocab size
        self.embedding = nn.Embedding(vocab_size,hidden_dim)

    def forward(self,token_ids: torch.tensor)-> torch.Tensor:
        tokens = self.embedding(token_ids)
        # Token shape: [Batch,16,hidden_dim]

        return tokens 
    

class StateEncoder(nn.Module):
    """
    Projects the 6 DOF robot joints state into the unified hidden dim 
    """

    def __init__(self,state_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(state_dim,hidden_dim)


    def forward(self,state: torch.Tensor) -> torch.Tensor:

        state_embedded = self.proj(state) #[Batch,hidden_dim]
        tokens = state_embedded.unsqueeze(1)

        return tokens 
    
