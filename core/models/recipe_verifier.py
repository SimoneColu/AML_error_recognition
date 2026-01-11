## transformer for entire recipe verification (extension)
## all recipe is correct or not

import torch
from torch import nn

from core.models.blocks import PositionalEncoding,EncoderLayer, Encoder, MLP, fetch_input_dim

class RecipeVerifier(nn.Module):
    def __init__(self,config,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.input_dim = 1024

        self.input_proj = nn.Sequential(
            nn.Linear(1792, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        ## positional encoding to keep sequence order
        self.positional_encoder = PositionalEncoding(d_model=self.input_dim,dropout=0.3,max_len=5000)

        ## transformer encoder
        step_encoder_layer = EncoderLayer(d_model=self.input_dim, dim_feedforward=2048, nhead=8, batch_first=True)
        self.step_encoder = Encoder(step_encoder_layer, num_layers=2)

        ## decoder (binary classification)
        self.decoder = MLP(self.input_dim, 512, 1)

    def forward(self,x,mask):
        # clean the input
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)   # (B,T,1792)

     
        x = self.input_proj(x)                                      # (B,T,1024)

        # add the positional encoder
        x = self.positional_encoder(x)                              # (B,T,1024)
        

        # Transformer Encoder
        # pass the mask (src_key_padding_mask) to ignore the padding
        x = self.step_encoder(x, src_key_padding_mask=mask)         # (B,T,1024)

        # --- MAX POOLING --- 
        # Vogliamo rilevare se c'è ALMENO un errore, quindi il segnale più forte vince.
        
        # 1. Gestione Padding per Max Pooling
        # Dobbiamo sostituire i vettori di padding con -infinito, 
        # altrimenti il max() potrebbe prendere uno 0.0 di padding invece di un valore negativo rilevante.
        
        # Espandiamo la maschera per adattarla alle feature: (B, T, 1024)
        # mask è True dove c'è padding.
        mask_expanded = mask.unsqueeze(-1).expand(x.size())
        
        # Riempiamo il padding con un valore molto basso (-1e9)
        x_masked = x.masked_fill(mask_expanded, -1e9)
        
        # 2. Global Max Pooling
        # Prendiamo il valore massimo su tutta la sequenza temporale (dim=1)
        x_max, _ = x_masked.max(dim=1)                  # (B, 1024)

        # 4. Binary Classification
        x = self.decoder(x_max)                                     # (B,1)

        return x



