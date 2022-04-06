from models import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import *
import logging
import time # debugging
from plot import *
from helpers import *
from joblib import load
from numpy import random
from icecream import ic
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def teacher_forcing(model_number, dataloader, EPOCH, lr, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device):

    device = torch.device(device)
    model_dic = {
        "CNN": CNN().double().to(device),
        "RNN": RNN().double().to(device),
        "GRU": GRU().double().to(device),
        "MLP_1": model_MLP_1().double().to(device),
        "LSTM_1": model_LSTM_1().double().to(device),
        "MLP_7": model_MLP_7().double().to(device),
        "LSTM_7": model_LSTM_7().double().to(device),
        "Attn_LSTM": AttentionalLSTM().double().to(device), 
        "Transformer": Transformer().double().to(device)
        }
    model_dic_keys = model_dic.keys()
    model_dic_keys_ls = list(model_dic_keys)
    madel_dic_values = model_dic.values()
    model_dic_values_ls = list(madel_dic_values)
    model = model_dic_values_ls[model_number]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):

        train_loss = 0
        val_loss = 0

        ## TRAIN -- TEACHER FORCING
        model.train() #setting the mode to train
        for index_in, index_tar, _input, target, index in dataloader: # for each data set 
        
            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]
            optimizer.zero_grad()
            #the permute here swap the position of input_length and batch
            src = _input.permute(1,0,2).double().to(device)[:-1,:,:] # torch.Size([24, 1, 7]) #save the last token for prediction
            target = _input.permute(1,0,2).double().to(device)[1:,:,:] # src shifted by 1.
            prediction = model(src, device) # torch.Size([24, 1, 7])
            loss = criterion(prediction, target[:,:,0].unsqueeze(-1)) #only calculate the loss of the predicted value
            loss.backward()
            optimizer.step()
            # scheduler.step(loss.detach().item())
            train_loss += loss.detach().item()

        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_to_save_model + f"{model_dic_keys_ls[model_number]}_best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"{model_dic_keys_ls[model_number]}_optimizer_{epoch}.pth")
            min_train_loss = train_loss
            best_model = f"{model_dic_keys_ls[model_number]}_best_train_{epoch}.pth"


        if epoch % 1 == 0: # Plot 1-Step Predictions

            # logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            scaler = load('scalar_item.joblib')
            src_ammonia = scaler.inverse_transform(src[:,:,0].cpu()) #torch.Size([35, 1, 7])
            target_ammonia = scaler.inverse_transform(target[:,:,0].cpu()) #torch.Size([35, 1, 7])
            prediction_ammonia = scaler.inverse_transform(prediction[:,:,0].detach().cpu().numpy()) #torch.Size([35, 1, 7])
            # plot_training(epoch, path_to_save_predictions, src_ammonia, prediction_ammonia, index_in, index_tar, model_dic_keys_ls[model_number])

        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)
        
    plot_loss(model_dic_keys_ls[model_number],path_to_save_loss, train=True)
    # print(f'Load inference with:{best_model}')
    # output = [min_train_loss, best_model]
    return min_train_loss,best_model

def flip_from_probability(p):
    return True if random.random() < p else False

def scheduled_sampling(model_number, dataloader, EPOCH, lr, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device, training_length):

    device = torch.device(device)
    model_dic = {
        "CNN": CNN().double().to(device),
        "RNN": RNN().double().to(device),
        "GRU": GRU().double().to(device),
        "MLP_1": model_MLP_1().double().to(device),
        "LSTM_1": model_LSTM_1().double().to(device),
        "MLP_7": model_MLP_7().double().to(device),
        "LSTM_7": model_LSTM_7().double().to(device),
        "Attn_LSTM": AttentionalLSTM().double().to(device), 
        "Transformer": Transformer().double().to(device)
        }

    model_dic_keys = model_dic.keys()
    model_dic_keys_ls = list(model_dic_keys)
    madel_dic_values = model_dic.values()
    model_dic_values_ls = list(madel_dic_values)
    model = model_dic_values_ls[model_number]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss = 0
        val_loss = 0

        ## TRAIN -- TEACHER FORCING
        model.train()
        for index_in, index_tar, _input, target, start in dataloader:
            #DataLoader automatically append new dim, which is batch size dim
            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]

            optimizer.zero_grad()
            src = _input.permute(1,0,2).double().to(device)[:-1,:,:] # torch.Size([24, 1, 7]) >> should be corrected to [47,1,7]
            target = _input.permute(1,0,2).double().to(device)[1:,:,:] # src shifted by 1. >> should be corrected to [47,1,7]
            sampled_src = src[:1, :, :] #t0 torch.Size([1, 1, 7])

            for i in range(len(target)-1): #i range from 0~45
                # the model output is the value of next timestep
                prediction = model(sampled_src, device) # torch.Size([1xw, 1, 1])
                # for p1, p2 in zip(params, model.parameters()):
                #     if p1.data.ne(p2.data).sum() > 0:
                #         ic(False)
                # ic(True)
                # ic(i, sampled_src[:,:,0], prediction)
                # time.sleep(1)
                """
                # to update model at every step
                # loss = criterion(prediction, target[:i+1,:,:1])
                # loss.backward()
                # optimizer.step()
                """
                #in the part, the predicted value after step 24 is randomly replaced by the gt at the predicted time step
                if i < training_length/2: # One day, enough data to make inferences about cycles
                    prob_true_val = True
                else:
                    ## coin flip
                    v = k/(k+math.exp(epoch/k)) # probability of heads/tails depends on the epoch, evolves with time.
                    prob_true_val = flip_from_probability(v) # starts with over 95 % probability of true val for each flip in epoch 0.
                    ## if using true value as new value
                if prob_true_val: # Using true value as next value #using detach to prevent other arithmetic operators included
                    # print(sampled_src.size())
                    # print(src.size())
                    # print(src[i+1, :, :].size())
                    # print(src[i+1, :, :].unsqueeze(0).size())
                    # exit()
                    sampled_src = torch.cat((sampled_src.detach(), src[i+1, :, :].unsqueeze(0).detach())) #for src[i+1,:,:], dim = 2, unsqueeze(0) will bring the dim bk t 3
                else: ## using prediction as new value
                    positional_encodings_new_val = src[i+1,:,1:].unsqueeze(0) # pos_en_n_v = [1, :, 6] # the prediction dim is [1,1,1]
                    predicted_ammonia = torch.cat((prediction[-1,:,:].unsqueeze(0), positional_encodings_new_val), dim=2) #[1,1,1] cat [1,1,6]
                    sampled_src = torch.cat((sampled_src.detach(), predicted_ammonia.detach()))
            # print(prediction.size()) #torch.Size([46, 1, 1])
            # exit()
            """To update model after each sequence"""
            # len(target[:-1,:,0]) = 46, 
            # print(prediction.size()) # torch.Size([46, 1, 1])
            # exit()
            loss = criterion(prediction,target[:-1,:,0].unsqueeze(-1)) #the order is reversed? need to check # unqueeze here is to make dim from 2 to 3 again
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item() #detach here is unneccessary?

        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_to_save_model + f"{model_dic_keys_ls[model_number]}_best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"{model_dic_keys_ls[model_number]}_optimizer_{epoch}.pth")
            min_train_loss = train_loss
            best_model = f"{model_dic_keys_ls[model_number]}_best_train_{epoch}.pth"

        if epoch % 1 == 0: # Plot 1-Step Predictions

            # logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            scaler = load('scalar_item.joblib')
            # print(sampled_src[:,:,0].size())
            # print(sampled_src[:,:,0])
            # print(sampled_src[:,0,0])
            # exit()
            sampled_src_ammonia = scaler.inverse_transform(sampled_src[:,:,0].cpu()) #torch.Size([47, 1, 7]) # the sampled input ammonia
            # print(sampled_src_ammonia)
            # print(np.shape(sampled_src_ammonia))
            # exit()
            src_ammonia = scaler.inverse_transform(src[:,:,0].cpu()) #torch.Size([47, 1, 7]) # the input ammonia (ground truth)
            target_ammonia = scaler.inverse_transform(target[:,:,0].cpu()) #torch.Size([47, 1, 7]) # the gt of predicted ammonia
            prediction_ammonia = scaler.inverse_transform(prediction[:,:,0].detach().cpu().numpy()) #torch.Size([47, 1, 7])
            plot_training_3(epoch, path_to_save_predictions, src_ammonia, sampled_src_ammonia, prediction_ammonia, index_in, index_tar)

        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)
        
    # plot_loss(model_dic_keys_ls[model_number],path_to_save_loss, train=True)
    # print(best_model)
    # output = [min_train_loss,best_model]
    return min_train_loss,best_model