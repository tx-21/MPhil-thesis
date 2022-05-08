import torch.nn as nn
import torch, math
from icecream import ic
import time

class Transformer(nn.Module):
    # d_model : number of features
    # only doing self-att, mask is required
    def __init__(self,feature_size=4,num_layers=1,dropout=0): # changed from 3 to 1 #stacking the decoder blocks three times
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, device): 
        x = src
        mask = self._generate_square_subsequent_mask(len(x)).to(device)
        output = self.transformer_encoder(x,mask)
        output = self.decoder(output)
        return output
