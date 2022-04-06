import argparse
from train import *
from DataLoader import *
from DataLoader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from helpers import *
from inference import *

def main(
    epoch: int = 400,
    k: int = 60,
    batch_size: int = 1,
    frequency: int = 100,
    training_length = 24,
    forecast_window = 1,
    Exp_num = 10,
    lr = 0.0003,
    train_csv = "train_dataset.csv",
    test_csv = "test_dataset.csv",
    path_to_save_model = "save_model/",
    path_to_save_loss_1 = "loss/save_loss_CNN/",
    path_to_save_loss_2 = "loss/save_loss_RNN/",
    path_to_save_loss_3 = "loss/save_loss_GRU/",
    path_to_save_loss_4 = "loss/save_loss_DNN/",
    path_to_save_loss_5 = "loss/save_loss_LSTM/",
    path_to_save_predictions_1 = "pred/save_predictions_CNN/",
    path_to_save_predictions_2 = "pred/save_predictions_RNN/",
    path_to_save_predictions_3 = "pred/save_predictions_GRU/",
    path_to_save_predictions_4 = "pred/save_predictions_DNN/",
    path_to_save_predictions_5 = "pred/save_predictions_LSTM/",
    device = 'cpu'
):

    metric_all = []
    Exp_num = Exp_num
    for m in range(1,9):
        database = m
        if database == 1:
            root_database = "data/train/obs/"
            result_loc = "obs"
            path_to_save_predictions_1 = result_loc + '/' + path_to_save_predictions_1
        if database == 2:
            root_database = "data/train/sg5/"        
        if database == 3:
            root_database = "data/train/sg7/"
        if database == 4:
            root_database = "data/train/sg9/"
        if database == 5:
            root_database = "data/train/ew2/"
        if database == 6:
            root_database = "data/train/ew3/"
        if database == 7:
            root_database = "data/train/ew4/"
        if database == 8:
            root_database = "data/train/or/"

        for j in range(Exp_num):
            clean_directory()

            train_dataset = TrainDataset(csv_name = train_csv, root_dir = root_database, training_length = training_length, forecast_window = forecast_window)
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            test_dataset = TestDataset(csv_name = test_csv, root_dir = "data/test/", training_length = training_length, forecast_window = forecast_window)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            #k was removed for using train_teacher_forcing
            model_number = 5
            #num1 = MLP_1, num2 = MLP_7, num3 = LSTM_1, num4 = LSTM_7, num5 = Transformer
            path_to_save_predictions = [path_to_save_predictions_1,path_to_save_predictions_2,path_to_save_predictions_3,path_to_save_predictions_4,path_to_save_predictions_5]
            path_to_save_loss = [path_to_save_loss_1,path_to_save_loss_2,path_to_save_loss_3,path_to_save_loss_4,path_to_save_loss_5]
            metric = []

            for i in range(model_number):
                train_loss, best_model = teacher_forcing(i, train_dataloader, epoch, lr, k, frequency, path_to_save_model, path_to_save_loss[i], path_to_save_predictions[i], device)
                # train_loss, best_model = scheduled_sampling(i, train_dataloader, epoch, lr, k, frequency, path_to_save_model, path_to_save_loss[i], path_to_save_predictions[i], device, training_length)
                _rmse, r2, val_loss = inference(i, path_to_save_predictions[i], forecast_window, test_dataloader, device, path_to_save_model, best_model, path_to_save_loss[i])
                status = []
                status = [_rmse, r2, val_loss, train_loss, best_model]
                metric.append(status)
        
            metric_all.append(metric)
            print(f'Exp_num {j+1} has finished')
    
        # metric = pd.DataFrame(metric, columns = ['fc_1_rmse','fc_2_rmse','fc_3_rmse','fc_1_r2','fc_2_r2','fc_3_r2','test_loss','train_loss','best_model'],index = ['MLP_1','MLP_7','LSTM_1','LSTM_7','Transformer'])
        # metric = pd.DataFrame(metric, columns = ['fc_1_rmse','fc_1_r2','test_loss','train_loss','best_model'],index = ['MLP_1','MLP_7','LSTM_1','LSTM_7','Transformer'])

        stability_test(metric_all, model_number, Exp_num, result_loc)

    # best_model = transformer(train_dataloader, epoch, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions[i], device)
