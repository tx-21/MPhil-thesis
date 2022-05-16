import torch.nn as nn
import torch, math
from icecream import ic
import time

class model_MLP_1(torch.nn.Module):
        def __init__(self, n_input=1, n_hidden=10, n_batch=1, n_output=1):
            super(model_MLP_1, self).__init__()
            self.input_size = n_input
            self.hidden_size  = n_hidden
            self.batch_size = n_batch
            self.output_size = n_output
            self.fc1 = nn.Linear(self.input_size, self.hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(self.hidden_size, self.output_size)
            
        def forward(self, src, device):
            output = self.fc1(src[:,:,0].unsqueeze(-1))
            output = self.relu(output)
            output = self.fc2(output)
            return output  

class model_MLP_7(torch.nn.Module):
        def __init__(self, n_input=7, n_hidden=10, n_batch=1, n_output=1):
            super(model_MLP_7, self).__init__()
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

class RNN_Model(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons, n_outputs):
        super(RNN_Model, self).__init__()
        self.batch_size=batch_size
        self.n_inputs=n_inputs
        self.n_neurons=n_neurons
        self.n_outputs=n_outputs
        self.rnn = nn.RNNCell(n_inputs, n_neurons)
        self.hx = torch.zeros(batch_size, n_neurons) # initialize hidden state
        self.FC = nn.Linear(n_neurons, n_outputs)
    def forward(self, X):
        self.hx = torch.zeros(self.batch_size, self.n_neurons)
        self.hx = self.rnn(X, self.hx)
        out = self.FC(self.hx)  
        out=out.reshape(out.size(1))
        return out

class model_LSTM_1(nn.Module):
    def __init__(self, n_hidden=21): #was 21 originally
        super(model_LSTM_1, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = 1
        self.lstm = nn.LSTM(input_size = 1, hidden_size = self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1) #predict 1-step

    def forward(self, src, device):
        h_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        c_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        h_t, c_t = self.lstm(src[:,:,0].unsqueeze(-1), (h_t,c_t))
        output = self.linear(h_t)
        return output

class model_LSTM_7(nn.Module):
    def __init__(self, n_hidden=21): #was 21 originally
        super(model_LSTM_7, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = 1
        self.lstm = nn.LSTM(input_size = 7, hidden_size = self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1) #predict 1-step

    def forward(self, src, device):
        h_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        c_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        x = src
        h_t, c_t = self.lstm(x, (h_t,c_t))
        output = self.linear(h_t)
        return output

class Transformer(nn.Module):
    # d_model : number of features
    # only doing self-att, mask is required
    def __init__(self,feature_size=7,num_layers=1,dropout=0): # changed from 3 to 1 #stacking the decoder blocks three times
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=7, dropout=dropout)
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

