import torch 
import torch.nn as nn 

from prana.model.encoders import LanguageEncoder, VisionEncoder, StateEncoder

class PranaVLA(nn.Module):
    """

    PRANA model 1
    """

    def __init__(
            self,
            action_dim: int = 6,
            state_dim: int = 6,
            chunk_size : int = 50,
            hidden_dim: int = 256,
            vocab_size: int = 256000,
    ):
        
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize encoders 
        self.vision_encoder = VisionEncoder(hidden_dim=hidden_dim)
        self.language_encoder = LanguageEncoder(vocab_size=vocab_size)
        self.state_encoder = StateEncoder(state_dim=state_dim,hidden_dim=hidden_dim)

        # Action queries: This ask what to do to the key pair from vision and language

        self.action_queries = nn.Parameter(torch.randn(1,chunk_size,hidden_dim))

        # The engine 

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout= 0.2,
            batch_first= True,
            norm_first= True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=4)


        # Action head 
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,action_dim)
        )


    def forward(self,image: torch.Tensor, state: torch.Tensor, lang_token: torch.Tensor)->torch.Tensor:

        if isinstance(image, list):
            
            batch_size = image[0].shape[0]
            
            all_v_tokens = []
            for img in image:
                all_v_tokens.append(self.vision_encoder(img)) 
            
            # Stitch patches together: [Batch, Patches * 2, Dim]
            v_tokens = torch.cat(all_v_tokens, dim=1) 
            
        else:
            # Fallback for a single image tensor
            batch_size = image.shape[0]
            v_tokens = self.vision_encoder(image)
  

        
        l_tokens = self.language_encoder(lang_token)
        s_tokens = self.state_encoder(state)


        # Expand the action query 

        q_tokens = self.action_queries.expand(batch_size,-1,-1)

        sequence = torch.cat([v_tokens,l_tokens,s_tokens,q_tokens],dim=1)

        # Pass through the engine 
        output_seq = self.transformer(sequence)
        # Extract only the updated action queries
        updated_queries = output_seq[:,-self.chunk_size:,:]

        # Project to 6 Dof robot actions 
        action_chunks = self.action_head(updated_queries)


        return action_chunks