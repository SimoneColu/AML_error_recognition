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
        self.input_dim = 256
        # Internal model dimension 
        self.internal_dim = 256

        # Projection of the input to the internal dimension
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.internal_dim),
            nn.LayerNorm(self.internal_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Positional encoding to keep sequence order
        self.positional_encoder = PositionalEncoding(
            d_model = self.internal_dim,
            dropout = self.config.dropout,
            max_len = 5000)

        # Transformer encoder
        step_encoder_layer = EncoderLayer(
            d_model = self.internal_dim, 
            dim_feedforward = 512,
            nhead = 4, 
            dropout = 0.5,
            batch_first = True)
        
        self.step_encoder = Encoder(
            step_encoder_layer, 
            num_layers=1)

        # Decoder (Binary Classification)
        # Input size doubled cause we test the Hybrid Pooling (Max + Avg)
        self.decoder = MLP(
            input_size = self.internal_dim * 2,
            hidden_size = 128, 
            output_size = 1)

    def forward(self,x,mask):
        # x shape: (Batch, Steps, 256)
        
        # clean the input
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)   # (B, T, input_dim)

        # Project to internal dimension
        x = self.input_proj(x)                                      # (B, T, internal_dim)

        # add the positional encoder
        x = self.positional_encoder(x)                              # (B,T,256)
        
        # Transformer Encoder
        x = self.step_encoder(x, src_key_padding_mask=mask)         # (B,T,256)

        # --- HYBRID POOLING (Max + Avg) --- 
        # Improves accuracy by capturing both peak errors and general context.
        
        # Mask Preparation
        mask_expanded = mask.unsqueeze(-1).expand(x.size())
        
        # Max Pooling (Detects strong error signals)
        x_masked_max = x.masked_fill(mask_expanded, -1e9)
        x_max, _ = x_masked_max.max(dim=1)                          # (B, internal_dim)
        
        # Average Pooling (Captures global context)
        x_masked_sum = x.masked_fill(mask_expanded, 0.0)
        x_sum = x_masked_sum.sum(dim=1)          # Sum over steps

        # Count non-padding steps to divide correctly
        # (~mask) gives 1 for real data, 0 for padding
        lengths = (~mask).sum(dim=1, keepdim=True).float() 
        x_avg = x_sum / lengths.clamp(min=1.0)                      # (B, internal_dim)

        # Concatenate
        x_cat = torch.cat([x_max, x_avg], dim=1)                    # (B, internal_dim * 2)

        # Binary Classification
        x = self.decoder(x_cat)                                     # (B,1)

        return x



