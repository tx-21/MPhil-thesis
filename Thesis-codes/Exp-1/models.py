import torch.nn as nn
import torch, math
from icecream import ic
import time

class CNN(nn.Module):
    """Convolutional Neural Networks"""
    def __init__(self, input_size=1, hidden_dim=1, output_size=1):
        super(CNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim, 10),
            nn.Linear(10, output_size)
        )

    def forward(self, src, device):
        # src = src.permute(1,0,2) #change from [input_lengh, batch, feature] to #[batch, input_length, feature]
        out = self.main(src[:,:,0].unsqueeze(-1))
        out = out.unsqueeze(1)
        # out = out.permute(1,0,2) #change back to the original format
        return out


class RNN(nn.Module):
    """Vanilla RNN"""
    def __init__(self, input_size=1, hidden_size=10, num_layers=1, output_size=1):
        super(RNN, self).__init__()

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
        out, _ = self.rnn(src[:,:,0].unsqueeze(-1), h_t)
        out = self.fc(out)
        return out

class GRU(nn.Module):
    """Gat e Recurrent Unit"""
    def __init__(self, input_size=1, hidden_size=10, num_layers=1, output_size=1):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, device):
        h_t = torch.zeros(self.num_layers,1,self.hidden_size).double()
        out, _ = self.gru(src[:,:,0].unsqueeze(-1),h_t)
        out = self.fc(out)
        return out


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