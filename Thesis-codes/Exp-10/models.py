import torch.nn as nn
import torch, math
from icecream import ic
import time

class LSTM_4(nn.Module):
    def __init__(self, n_hidden=21): #was 21 originally
        super(LSTM_4, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = 1
        self.lstm = nn.LSTM(input_size = 4, hidden_size = self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 2) #predict 1-step

    def forward(self, src, device):
        h_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        c_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        x = src
        h_t, c_t = self.lstm(x, (h_t,c_t))
        output = self.linear(h_t)
        return output