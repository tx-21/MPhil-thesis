import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import dump
from icecream import ic

class TrainDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_name, root_dir, training_length, forecast_window):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """
        
        # load raw data file
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = MinMaxScaler()
        self.T = training_length
        self.S = forecast_window

    def __len__(self):
        # return number of sensors
        return len(self.df)

    # Will pull an index between 0 and __len__. 
    def __getitem__(self, index): #idx will iterate between 0 and __len__
        

        # np.random.seed(0)
        #from codeline 43 to 49, is to take only 1 sample, which contains the training and testing only within one sensor (idx is assigned in the arg)
    
        #there might have a small error here, which len(df) - T - S does not include the last sample (can still run but not sure the influence of it)
        start = np.random.randint(0, len(self.df) - self.T - self.S + 1)  
        index_in = torch.tensor([i for i in range(start, start+self.T)])
        index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])
        _input = torch.tensor(self.df[["Colour", "sin_hour", "cos_hour", "sin_day", "cos_day"]][start : start + self.T].values)
        target = torch.tensor(self.df[["Colour", "sin_hour", "cos_hour", "sin_day", "cos_day"]][start + self.T : start + self.T + self.S].values)

        # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
        # scalar is fit only for humidity, as the timestamps are already scaled
        # scalar input/output of shape: [n_samples, n_features].
        scaler = self.transform
        # print(_input[:,0].size()) # torch.Size([48])_inpu
        # print(_input[:,0].unsqueeze(-1).size())
        # print(_input[:,0].unsqueeze(-1).squeeze(-1).size())
        # # exit()
        scaler.fit(_input[:,0].unsqueeze(-1)) #the minmaxscaler can only be applied on arr with dim>=2
        _input[:,0] = torch.tensor(scaler.transform(_input[:,0].unsqueeze(-1)).squeeze(-1))
        target[:,0] = torch.tensor(scaler.transform(target[:,0].unsqueeze(-1)).squeeze(-1))
        # save the scalar to be used later when inverse translating the data for plotting.
        dump(scaler, 'scalar_item.joblib')

        return index_in, index_tar, _input, target, start

class TestDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_name, root_dir, training_length, forecast_window):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """
        
        # load raw data file
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = MinMaxScaler()
        self.T = training_length
        self.S = forecast_window

    def __len__(self):
        return 24*5 #the length of the test dataset

    # Will pull an index between 0 and __len__. 
    def __getitem__(self, index): #idx will iterate between 0 and __len__
        #only need len(df) - self.T - self.S + 1, but index is len(df)
        #from codeline 43 to 49, is to take only 1 sample, which contains the training and testing only within one sensor (idx is assigned in the arg)
        #there might have a small error here, which len(df) - T - S does not include the last sample (can still run but not sure the influence of it)
        _len = len(self.df) - self.T - self.S + 1
        _list = np.arange(_len - 24*5, _len) #last 7 days is no.506 ~ no.697, input = last 8 ~ last 1, output = last 7 ~ last
        start = _list[index] # index len = 10
        # start = index - self.T - self.S
        index_in = torch.tensor([i for i in range(start, start+self.T)])
        index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])
        _input = torch.tensor(self.df[["Colour", "sin_hour", "cos_hour", "sin_day", "cos_day"]][start : start + self.T].values)
        target = torch.tensor(self.df[["Colour", "sin_hour", "cos_hour", "sin_day", "cos_day"]][start + self.T : start + self.T + self.S].values)

        # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
        # scalar is fit only for humidity, as the timestamps are already scaled
        # scalar input/output of shape: [n_samples, n_features].
        scaler = self.transform
        # print(_input[:,0].size()) # torch.Size([48])_inpu
        # print(_input[:,0].unsqueeze(-1).size())
        # print(_input[:,0].unsqueeze(-1).squeeze(-1).size())
        # # exit()
        scaler.fit(_input[:,0].unsqueeze(-1)) #the minmaxscaler can only be applied on arr with dim>=2
        _input[:,0] = torch.tensor(scaler.transform(_input[:,0].unsqueeze(-1)).squeeze(-1))
        target[:,0] = torch.tensor(scaler.transform(target[:,0].unsqueeze(-1)).squeeze(-1))
        # save the scalar to be used later when inverse translating the data for plotting.
        dump(scaler, 'scalar_item.joblib')

        return index_in, index_tar, _input, target, start
