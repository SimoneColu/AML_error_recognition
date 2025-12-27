## transformer for entire recipe verification (extension)
## all recipe is correct or not

import torch
from torch import nn

from core.models.blocks import PositionalEncoding,EncoderLayer, Encoder, MLP, fetch_input_dim

class RecipeVerifier(nn.Module):
    def __init__(self,config,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        ## retrieve feature dimensions
        input_dimension = fetch_input_dim(config)

        ## positional encoding to keep sequence order
        self.positional_encoder = PositionalEncoding(d_model=input_dimension,dropout=0.1,max_len=5000)

        ## transformer encoder
        step_encoder_layer = EncoderLayer(d_model=input_dimension, dim_feedforward=2048, nhead=8, batch_first=True)
        self.step_encoder = Encoder(step_encoder_layer, num_layers=1)

        ## decoder (binary classification)
        self.decoder = MLP(input_dimension, 512, 1)

    def forward(self,x,mask):
        # clean the input
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # add the positional encoder
        x = self.positional_encoder(x)
        

        # Transformer Encoder
        # pass the mask (src_key_padding_mask) to ignore the padding
        x = self.step_encoder(x, src_key_padding_mask=mask)

        # 3. Global Average Pooling (Masked)
        # Non possiamo fare semplicemente x.mean(dim=1) perché includerebbe gli zeri del padding nella media.
        
        # Invertiamo la maschera se necessario: ci serve 1 dove c'è dato, 0 dove c'è padding.
        # Assumendo che 'mask' sia True per il Padding (standard PyTorch):
        input_mask_expanded = (~mask).unsqueeze(-1).expand(x.size()).float()
        
        # Somma solo i vettori validi
        sum_embeddings = (x * input_mask_expanded).sum(1)
        
        # Conta quanti step validi ci sono per ogni ricetta
        sum_mask = input_mask_expanded.sum(1)
        
        # Evita divisione per zero (clamp)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        # Media corretta
        x = sum_embeddings / sum_mask

        # 4. Binary Classification
        x = self.decoder(x)



