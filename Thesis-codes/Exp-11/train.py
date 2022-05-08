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

def teacher_forcing(model_number, dataloader, valdataloader, forecast_window, EPOCH, lr, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device, current_exp, last_exp_num, scheduler_status, factor, patience):

    device = torch.device(device)
    model_dic = {
        "Attn-LSTM": Attn_LSTM().double().to(device),
        "Transformer": Transformer().double().to(device)
        }
    model_dic_keys = model_dic.keys()
    model_dic_keys_ls = list(model_dic_keys)
    madel_dic_values = model_dic.values()
    model_dic_values_ls = list(madel_dic_values)
    model = model_dic_values_ls[model_number]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')
    min_valid_loss = float('inf')

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
            loss = criterion(prediction[:,:,1].unsqueeze(-1), target[:,:,1].unsqueeze(-1)) #only calculate the loss of the predicted value
            loss.backward()
            optimizer.step()    
            # scheduler.step(loss.detach().item())
            train_loss += loss.detach().item()

        model.eval()
        for index_in, index_tar, _input, target, index in valdataloader:
            
            src = _input.permute(1,0,2).double().to(device)[:-1,:,:] # torch.Size([24, 1, 7]) #save the last token for prediction
            target = _input.permute(1,0,2).double().to(device)[1:,:,:] # src shifted by 1.
            prediction = model(src, device) # torch.Size([24, 1, 7])
            loss = criterion(prediction[:,:,1].unsqueeze(-1), target[:,:,1].unsqueeze(-1)) #originally it was the opposite
            val_loss += loss.detach().item()

        if scheduler_status:
            scheduler.step(val_loss)

        if min_valid_loss > val_loss:
            # print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
            os.makedirs(os.path.dirname(path_to_save_model), exist_ok=True)
            torch.save(model.state_dict(), path_to_save_model + f"{model_dic_keys_ls[model_number]}_best_train.pth")
            # torch.save(optimizer.state_dict(), path_to_save_model + f"{model_dic_keys_ls[model_number]}_optimizer_{epoch}.pth")
            best_model = f"{model_dic_keys_ls[model_number]}_best_train.pth"
            min_valid_loss = val_loss

        if min_train_loss > train_loss:
            min_train_loss = train_loss
            
        if epoch % 1 == 0: # Plot 1-Step Predictions

            # logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            scaler = load('scalar_item.joblib')
            src_ammonia = scaler.inverse_transform(src[:,:,:2].squeeze(1).cpu()) #torch.Size([35, 1, 7])
            # prediction_ammonia = scaler.inverse_transform(prediction[:,:,0].detach().cpu().numpy()) #torch.Size([35, 1, 7])
            src_ammonia = src_ammonia[:,1]
            target_ammonia = scaler.inverse_transform(target[:,:,:2].squeeze(1).cpu()) #torch.Size([35, 1, 7])
            prediction_ammonia = scaler.inverse_transform(prediction[:,:,:2].squeeze(1).detach().cpu().numpy()) #torch.Size([35, 1, 7])
            prediction_ammonia = prediction_ammonia[:,1]
            plot_training(epoch, path_to_save_predictions, src_ammonia, prediction_ammonia, index_in, index_tar, model_dic_keys_ls[model_number])

        if current_exp == last_exp_num:
            train_loss /= len(dataloader)
            val_loss /= len(dataloader)
            log_loss(train_loss, path_to_save_loss, train=True)
            log_loss(val_loss, path_to_save_loss, train=False)
        
        # if epoch == EPOCH and current_exp == last_exp_num: 
        #     print(f"{model_dic_keys_ls[model_number]}, lr at %d epoch：%f" % (epoch, optimizer.param_groups[0]['lr']))

    if current_exp == last_exp_num:
        plot_loss(model_dic_keys_ls[model_number],path_to_save_loss, train=True)
    if epoch == EPOCH:
        epoch_out = float(optimizer.param_groups[0]['lr'])
    # print(f'Load inference with:{best_model}')
    # output = [min_train_loss, best_model]
    return min_train_loss, min_valid_loss, best_model, epoch_out

def flip_from_probability(p):
    return True if random.random() < p else False