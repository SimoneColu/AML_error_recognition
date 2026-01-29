## transformer for entire recipe verification (extension)
## all recipe is correct or not

import torch
from torch import nn

from core.models.blocks import PositionalEncoding,EncoderLayer, Encoder, MLP, fetch_input_dim

class RecipeVerifier(nn.Module):
    def __init__(self,config,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        # Configurable Dimensions
        # Raw Features from step segmentation (S,256)
        self.input_dim = 768
        # Internal model dimension 
        self.internal_dim = 64


        # Projection of the input to the internal dimension
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.internal_dim),
            nn.LayerNorm(self.internal_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )

        # Positional encoding to keep sequence order
        self.positional_encoder = PositionalEncoding(
            d_model = self.internal_dim,
            dropout = self.config.dropout,
            max_len = 5000)

        # Transformer encoder
        step_encoder_layer = EncoderLayer(
            d_model = self.internal_dim, 
            dim_feedforward = 128,
            nhead = 2, 
            dropout = self.config.dropout,
            batch_first = True)
        
        self.step_encoder = Encoder(
            step_encoder_layer, 
            num_layers=1)

        # Decoder (Binary Classification)
        # Input size doubled cause we test the Hybrid Pooling (Max + Avg)
        self.decoder = MLP(
            input_size = self.internal_dim*2,
            hidden_size = 64, 
            output_size = 1)

    def forward(self,x,mask):
        # x shape: (Batch, Steps, 256)

        # --- Input Noise Injection ---
        if self.training:
            # Add random jitter. 
            noise = torch.randn_like(x) * 0.1  
            x = x + noise
        # -------------------------------------------
        
        # clean the input
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)   # (B, T, input_dim)

        # Project to internal dimension
        x = self.input_proj(x)                                      # (B, T, internal_dim)

        # add the positional encoder
        x = self.positional_encoder(x)                              # (B,T,256)
        
        # Transformer Encoder
        x = self.step_encoder(x, src_key_padding_mask=mask)         # (B,T,256)

        
        """
        # --- HYBRID POOLING ---
        
        # 1. Max Pooling
        mask_expanded = mask.unsqueeze(-1).expand(x.size())
        x_masked_max = x.masked_fill(mask_expanded, -1e9)
        x_max, _ = x_masked_max.max(dim=1)  # (B, 128)

        # 2. Average Pooling 
        # Invert mask: True for Data, False for Padding
        valid_mask_float = (~mask).unsqueeze(-1).float() 
        sum_embeddings = torch.sum(x * valid_mask_float, dim=1)
        sum_mask = torch.clamp(valid_mask_float.sum(dim=1), min=1e-9)
        x_avg = sum_embeddings / sum_mask   # (B, 128)

        # 3. CONCATENATE
        x_cat = torch.cat((x_max, x_avg), dim=1) # (B, 256)

        # Classify
        x = self.decoder(x_cat)
        return x 
        
       
        """
        
       # --- ONLY MAX POOLING --- 
        
        # Mask Preparation
        mask_expanded = mask.unsqueeze(-1).expand(x.size())
        
        # Max Pooling (Detects strong error signals)
        x_masked_max = x.masked_fill(mask_expanded, -1e9)
        x_max, _ = x_masked_max.max(dim=1)                          # (B, internal_dim)

        # Binary Classification
        # We pass x_max directly (size is internal_dim, not internal_dim * 2)
        x = self.decoder(x_max)                                     # (B,1)
        return x 
        
        """
        
        
        
        
        
        # --- ONLY AVERAGE POOLING ---

        # Invert mask: True for Data, False for Padding
        valid_mask_float = (~mask).unsqueeze(-1).float() 
        sum_embeddings = torch.sum(x * valid_mask_float, dim=1)
        sum_mask = torch.clamp(valid_mask_float.sum(dim=1), min=1e-9)
        x_avg = sum_embeddings / sum_mask   # (B, internal_dim)

        # Classify
        x = self.decoder(x_avg)
        return x
        
        """
        



