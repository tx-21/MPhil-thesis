import torch.nn as nn
import torch, math
from icecream import ic
import time

class LSTM_4(nn.Module):
    def __init__(self, n_hidden=21): #was 21 originally
        super(LSTM_4, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = 1
        self.out_1 = 1
        self.out_2 = 49
        self.out_3 = self.out_1 + self.out_2

        self.g1 = nn.LSTM(input_size = 3, hidden_size = self.n_hidden)
        self.linear1 = nn.Linear(self.n_hidden, self.out_1) #predict 1-step
        self.g2 = nn.LSTM(input_size = 3, hidden_size = self.n_hidden)
        self.linear2 = nn.Linear(self.n_hidden, self.out_2) #predict 1-step
        self.lstm = nn.LSTM(input_size = self.out_3, hidden_size = self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 2) #predict 1-step

    def forward(self, src, device):

        h_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        c_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        # src [23,1,4]
        x1 = src[:,:,0].unsqueeze(-1)
        x2 = src[:,:,1].unsqueeze(-1)        
        x3 = src[:,:,2:]

        input_1 = torch.cat((x1,x3),2) # [23,1,3]
        # input_1 = x1
        input_2 = torch.cat((x2,x3),2) # [23,1,3]
        h_t1, c_t1 = self.g1(input_1, (h_t,c_t)) # [23,1,1]
        h_t2, c_t2 = self.g2(input_2, (h_t,c_t))
        h_t1_out = self.linear1(h_t1) # [23,1,5]
        h_t2_out = self.linear2(h_t2) # [23,1,5]
        input_3 = torch.cat((h_t1_out, h_t2_out),2) # [23,1,10]
        h_t, c_t = self.lstm(input_3, (h_t,c_t))
        output = self.linear(h_t)
        return output
