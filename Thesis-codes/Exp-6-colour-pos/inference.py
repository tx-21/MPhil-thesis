from model_attn import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import *
import logging
import time # debugging
from plot import *
from helpers import *
from joblib import load
from icecream import ic
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)


def inference(model_number, path_to_save_predictions, forecast_window, dataloader, device, path_to_save_model, best_model,path_to_save_loss,current_exp,last_exp_num):

    device = torch.device(device)
    model_dic = {
        "RNN": RNN_3().double().to(device),
        "GRU": GRU_3().double().to(device),
        "LSTM_3": LSTM_3().double().to(device),
        "RNN_attn": RNN_attn().double().to(device),
        "GRU_attn": GRU_attn().double().to(device),
        "LSTM_5_attn": LSTM_attn().double().to(device),
        "Transformer": Transformer().double().to(device),
        "GRU_1": GRU().double().to(device),
        "LSTM_1": model_LSTM_1().double().to(device),
        "RNN_1": RNN().double().to(device)
        }
    model_dic_keys = model_dic.keys()
    model_dic_keys_ls = list(model_dic_keys)
    madel_dic_values = model_dic.values()
    model_dic_values_ls = list(madel_dic_values)
    model = model_dic_values_ls[model_number]
    model.load_state_dict(torch.load(path_to_save_model+best_model))
    criterion = torch.nn.MSELoss()

    val_loss = 0
    with torch.no_grad():

        model.eval()
        forecast_horizon = pd.DataFrame(columns = ['fh1','fh2','fh3'])
        fh1,fh2,fh3=[],[],[]
        fh1_true,fh2_true,fh3_true = [],[],[]
        for index_in, index_tar, _input, target, index in dataloader: #iterate all the samples
            # starting from 1 so that src matches with target (so that the first and the last predicted token match with the target), but has same length as when training
            src = _input.permute(1,0,2).double().to(device)[1:, :, :] # 47, 1, 7: t1 -- t47
            target = target.permute(1,0,2).double().to(device) # t48 - t59
            next_input_model = src
            all_predictions = []
            for i in range(forecast_window): # >> originally was forecast_window - 1
                
                prediction = model(next_input_model, device) # 47,1,1: t2' - t48'
                if all_predictions == []:
                    all_predictions = prediction # 47,1,1: t2' - t48'
                else:
                    all_predictions = torch.cat((all_predictions, prediction[-1,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'
                pos_encoding_old_vals = src[i + 1:, :, 1:] # 46, 1, 6, pop positional encoding first value: t2 -- t47
                pos_encoding_new_val = target[i, :, 1:].unsqueeze(1) # 1, 1, 6, append positional encoding of last predicted value: t48 >> originally was target[i+1,:,1:]
                pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6 positional encodings matched with prediction: t2 -- t48
                
                next_input_model = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1,:,:].unsqueeze(0))) #t2 -- t47, t48'
                next_input_model = torch.cat((next_input_model, pos_encodings), dim = 2) # 47, 1, 7 input for next round
            true = torch.cat((src[1:,:,0],target[:,:,0]))
            loss = criterion(all_predictions[:,:,0],true) #originally it was the opposite
            val_loss += loss.detach().item()
            
        
            scaler = load('scalar_item.joblib')
            src_ammonia = scaler.inverse_transform(src[:,:,0].cpu())
        
            target_ammonia = scaler.inverse_transform(target[:,:,0].cpu())
            # Extract the tru ammonia values 
            out_target_ammonia = target_ammonia.flatten()
            fh1_true.append(out_target_ammonia[0])
            fh2_true.append(out_target_ammonia[1])
            fh3_true.append(out_target_ammonia[2])
           
            prediction_ammonia = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy()) #np.shape = [25,1]   
            # Extract the ammonia values from the respective forecast horizon
            out_prediction_ammonia = prediction_ammonia.flatten()
            # print(np.shape(out_prediction_ammonia))
            # print(out_prediction_ammonia[-1])
            # fh1.append(out_prediction_ammonia[-1])
            fh1.append(out_prediction_ammonia[-3])
            fh2.append(out_prediction_ammonia[-2])
            fh3.append(out_prediction_ammonia[-1])
        
            # if current_exp == last_exp_num:
            #    plot_prediction(index, path_to_save_predictions, src_ammonia, target_ammonia, prediction_ammonia, index_in, index_tar)
    
        forecast_horizon['fh1'] = fh1
        forecast_horizon['fh2'] = fh2
        forecast_horizon['fh3'] = fh3
        forecast_horizon['fh1_true'] = fh1_true
        forecast_horizon['fh2_true'] = fh2_true
        forecast_horizon['fh3_true'] = fh3_true       
        # date_range = ['11-27','11-28','11-29','11-30']
        
        date_range = ['1-17','1-18','1-19','1-20','1-21','1-22','1-23']
        rmse_1, r2_1 = plot_prediction_horizon(forecast_horizon['fh1'], forecast_horizon['fh1_true'], '1', model_dic_keys_ls[model_number], path_to_save_predictions,date_range,current_exp,last_exp_num)
        rmse_2, r2_2 = plot_prediction_horizon(forecast_horizon['fh2'], forecast_horizon['fh2_true'], '2', model_dic_keys_ls[model_number], path_to_save_predictions,date_range,current_exp,last_exp_num)
        rmse_3, r2_3 = plot_prediction_horizon(forecast_horizon['fh3'], forecast_horizon['fh3_true'], '3', model_dic_keys_ls[model_number], path_to_save_predictions,date_range,current_exp,last_exp_num)
        
        val_loss /= len(dataloader)
        if current_exp == last_exp_num:
            log_test_loss(val_loss, path_to_save_loss)
            fc1 = np.column_stack([forecast_horizon['fh1'], forecast_horizon['fh1_true']])
            fc2 = np.column_stack([forecast_horizon['fh2'], forecast_horizon['fh2_true']])
            fc3 = np.column_stack([forecast_horizon['fh3'], forecast_horizon['fh3_true']])
            log_test_raw(fc1,path_to_save_loss,'fc1')
            log_test_raw(fc2,path_to_save_loss,'fc2')
            log_test_raw(fc3,path_to_save_loss,'fc3')     
                   
        logger.info(f"{model_dic_keys_ls[model_number]}_Loss On Unseen Dataset: {val_loss}")
        # metric_out = [rmse_1, rmse_2, rmse_3, r2_1, r2_2, r2_3, val_loss]
        # metric_out = [rmse_1, r2_1, val_loss]
    
    return rmse_1, r2_1, rmse_2, r2_2, rmse_3, r2_3, val_loss