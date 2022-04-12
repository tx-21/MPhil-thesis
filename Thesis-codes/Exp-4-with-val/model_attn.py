import torch.nn as nn
import torch, math
from icecream import ic
import time

# required revision on the attn lstm
class RNN_5(nn.Module):
    """Vanilla RNN"""
    def __init__(self, input_size=5, hidden_size=10, num_layers=1, output_size=1):
        super(RNN_5, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, device):
        h_t = torch.zeros(self.num_layers,1,self.hidden_size).double()
        out, _ = self.rnn(src, h_t)
        out = self.fc(out)
        return out

class model_MLP_5(torch.nn.Module):
    def __init__(self, n_input=5, n_hidden=10, n_batch=1, n_output=1):
        super(model_MLP_5, self).__init__()
        self.input_size = n_input
        self.hidden_size  = n_hidden
        self.batch_size = n_batch
        self.output_size = n_output
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
            
    def forward(self, src, device):
        output = self.fc1(src)
        output = self.relu(output)
        output = self.fc2(output)
        return output  

class model_LSTM_5(nn.Module):
    def __init__(self, n_hidden=21): #was 21 originally
        super(model_LSTM_5, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = 1
        self.lstm = nn.LSTM(input_size = 5, hidden_size = self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1) #predict 1-step

    def forward(self, src, device):
        h_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        c_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        x = src
        h_t, c_t = self.lstm(x, (h_t,c_t))
        output = self.linear(h_t)
        return output

class RNN_attn(nn.Module):
    """LSTM with Attention"""
    def __init__(self, input_size=5, qkv=5, hidden_size=10, num_layers=1, output_size=1, bidirectional=False):
        super(RNN_attn, self).__init__()

        self.input_size = input_size
        self.qkv = qkv
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.query = nn.Linear(input_size, qkv)
        self.key = nn.Linear(input_size, qkv)
        self.value = nn.Linear(input_size, qkv)

        self.attn = nn.Linear(qkv, input_size)
        self.scale = math.sqrt(qkv)

        # self.lstm = nn.LSTM(input_size=input_size,
        #                     hidden_size=hidden_size,
        #                     num_layers=num_layers,
        #                     batch_first=True,
        #                     bidirectional=bidirectional)

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, device):
        x = src
        Q, K, V = self.query(x), self.key(x), self.value(x)

        dot_product = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale
        scores = torch.softmax(dot_product, dim=-1)
        scaled_x = torch.matmul(scores, V) + x

        out = self.attn(scaled_x) + x
        out, _ = self.rnn(out)
        # out = out[:, -1, :]
        out = self.fc(out)

        return out

class Transformer(nn.Module):
    # d_model : number of features
    # only doing self-att, mask is required
    def __init__(self,feature_size=5,num_layers=1,dropout=0): # changed from 3 to 1 #stacking the decoder blocks three times
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=5, dropout=dropout)
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
