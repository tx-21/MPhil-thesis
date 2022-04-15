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
    forecast_window = 3,
    Exp_num = 10,
    lr = 0.0003,
    factor = 0.5,
    patience = 5,
    num_dataset: int = 9,
    scheduler_status = False,
    train_csv = "train_dataset.csv",
    # test_csv = "test_1010_1016_dataset.csv",
    test_csv = "test_1022_1028_dataset.csv",
    valid_csv = "valid_dataset.csv",
    path_to_save_model = "save_model/",
    path_to_save_loss_1 = "loss/save_loss_GRU/",
    path_to_save_loss_2 = "loss/save_loss_RNN/",
    path_to_save_loss_3 = "loss/save_loss_RNN_attn/",
    path_to_save_loss_4 = "loss/save_loss_Transformer/",
    path_to_save_loss_5 = "loss/save_loss_Transformer/",
    path_to_save_loss_6 = "loss/save_loss_Transformer/",
    path_to_save_loss_7 = "loss/save_loss_Transformer/",
    path_to_save_loss_8 = "loss/save_loss_Transformer/",
    path_to_save_loss_9 = "loss/save_loss_Transformer/",
    path_to_save_loss_10 = "loss/save_loss_Transformer/",
    path_to_save_predictions_1 = "pred/save_predictions_RNN/",
    path_to_save_predictions_2 = "pred/save_predictions_DNN/",
    path_to_save_predictions_3 = "pred/save_predictions_RNN_attn/",
    path_to_save_predictions_4 = "pred/save_predictions_Transformer/",
    path_to_save_predictions_5 = "pred/save_predictions_Transformer/",
    path_to_save_predictions_6 = "pred/save_predictions_Transformer/",
    path_to_save_predictions_7 = "pred/save_predictions_Transformer/",
    path_to_save_predictions_8 = "pred/save_predictions_Transformer/",
    path_to_save_predictions_9 = "pred/save_predictions_Transformer/",
    path_to_save_predictions_10 = "pred/save_predictions_Transformer/",
    device = 'cpu'
):

    Exp_num = Exp_num
    clean_directory()
    fc1_all_dataset = []
    fc2_all_dataset = []
    fc3_all_dataset = []
    all_model_dataset_name = []
    for m in range(1,num_dataset+1):
        metric_all = []
        database = m
        if database == 1:
            root_database = "data/train/obs/"
            result_loc = "results/LSTM-3-obs"
            dataset_name = 'LSTM-3-obs'
            pred_loc = 'LSTM-3-obs'
            model_num = 2 # GRU
        if database == 2:
            root_database = "data/train/sg5/"
            result_loc = "results/LSTM-3-sg5"
            dataset_name = 'LSTM-3-sg5'
            pred_loc = 'LSTM-3-sg5'
            model_num = 2 # RNN        
        if database == 3:
            root_database = "data/train/sg7/"
            result_loc = "results/LSTM-3-sg7"
            dataset_name = 'LSTM-3-sg7'
            pred_loc = 'LSTM-3-sg7'
            model_num = 2 # LSTM
        if database == 4: #anchor purpose
            root_database = "data/train/sg9/"
            result_loc = "results/LSTM-3-sg9"
            dataset_name = 'LSTM-3-sg9'
            pred_loc = 'LSTM-3-sg9'
            model_num = 2 
        if database == 5:
            root_database = "data/train/ew2/"
            result_loc = "results/LSTM-3-ew2"
            dataset_name = 'LSTM-3-ew2'
            pred_loc = 'LSTM-3-ew2'
            model_num = 2 
        if database == 6:
            root_database = "data/train/ew3/"
            result_loc = "results/LSTM-3-ew3"
            dataset_name = 'LSTM-3-ew3'
            pred_loc = 'LSTM-3-ew3'
            model_num = 2
        if database == 7:
            root_database = "data/train/ew4/"
            result_loc = "results/LSTM-3-ew4"
            dataset_name = 'LSTM-3-ew4'
            pred_loc = 'LSTM-3-ew4'
            model_num = 2 
        if database == 8:
            root_database = "data/train/or/"
            result_loc = "results/LSTM-3-or"
            dataset_name = 'LSTM-3-or'
            pred_loc = 'LSTM-3-or'
            model_num = 2
        if database == 9:
            root_database = "data/train/sg7/"
            result_loc = "results/RNN-anchor"
            dataset_name = 'RNN-anchor'
            pred_loc = 'RNN-anchor/'
            model_num = 9
        if database == 10:
            root_database = "data/train/sg7/"
            result_loc = "results/Transformer"
            dataset_name = 'Transformer'
            pred_loc = 'Transformer/'
            model_num = 6

                    
        path_to_save_model_new = "results-archive/result-5-0415-0116-0122/" + pred_loc + "/" + path_to_save_model
        path_to_save_loss_1_new = result_loc + '/' + path_to_save_loss_1
        path_to_save_loss_2_new = result_loc + '/' + path_to_save_loss_2
        path_to_save_loss_3_new = result_loc + '/' + path_to_save_loss_3
        path_to_save_loss_4_new = result_loc + '/' + path_to_save_loss_4
        path_to_save_loss_5_new = result_loc + '/' + path_to_save_loss_5
        path_to_save_loss_6_new = result_loc + '/' + path_to_save_loss_6
        path_to_save_loss_7_new = result_loc + '/' + path_to_save_loss_7
        path_to_save_loss_8_new = result_loc + '/' + path_to_save_loss_8
        path_to_save_loss_9_new = result_loc + '/' + path_to_save_loss_9
        path_to_save_loss_10_new = result_loc + '/' + path_to_save_loss_10

        path_to_save_predictions_1_new = result_loc + '/' + path_to_save_predictions_1
        path_to_save_predictions_2_new = result_loc + '/' + path_to_save_predictions_2
        path_to_save_predictions_3_new = result_loc + '/' + path_to_save_predictions_3
        path_to_save_predictions_4_new = result_loc + '/' + path_to_save_predictions_4
        path_to_save_predictions_5_new = result_loc + '/' + path_to_save_predictions_5
        path_to_save_predictions_6_new = result_loc + '/' + path_to_save_predictions_6
        path_to_save_predictions_7_new = result_loc + '/' + path_to_save_predictions_7
        path_to_save_predictions_8_new = result_loc + '/' + path_to_save_predictions_8
        path_to_save_predictions_9_new = result_loc + '/' + path_to_save_predictions_9
        path_to_save_predictions_10_new = result_loc + '/' + path_to_save_predictions_10

        metric = []
        for j in range(Exp_num):
            current_exp = j
            last_exp_num = Exp_num-1
            path_model_exp = path_to_save_model_new + f"{current_exp}/"
            # train_dataset = TrainDataset(
            #     csv_name = train_csv, root_dir = root_database, training_length = training_length,
            #     forecast_window = forecast_window
            #     )
            # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            # val_dataset = TrainDataset(
            #     csv_name = valid_csv, root_dir = "data/valid/", training_length = training_length,
            #     forecast_window = forecast_window
            #     )
            # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
            test_dataset = TestDataset(
                csv_name = test_csv, root_dir = "data/test/", training_length = training_length,
                forecast_window = forecast_window
                )
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            #k was removed for using train_teacher_forcing
            path_to_save_predictions = [
                path_to_save_predictions_1_new,path_to_save_predictions_2_new,path_to_save_predictions_3_new,path_to_save_predictions_4_new,
                path_to_save_predictions_5_new,path_to_save_predictions_6_new,path_to_save_predictions_7_new,path_to_save_predictions_8_new,path_to_save_predictions_9_new,path_to_save_predictions_10_new
                ]
            path_to_save_loss = [
                path_to_save_loss_1_new,path_to_save_loss_2_new,path_to_save_loss_3_new,path_to_save_loss_4_new,
                path_to_save_loss_5_new,path_to_save_loss_6_new,path_to_save_loss_7_new,path_to_save_loss_8_new,path_to_save_loss_9_new,path_to_save_loss_10_new
                ]
            
            # train_loss, valid_loss, best_model, epoch_out = teacher_forcing(
            #     model_num, train_dataloader,val_dataloader, forecast_window, epoch, lr, k, frequency,
            #     path_model_exp, path_to_save_loss[database-1],path_to_save_predictions[database-1], 
            #     device, current_exp, last_exp_num, scheduler_status, factor, patience
            #     )
            # train_loss, best_model = scheduled_sampling(i, train_dataloader, epoch, lr, k, frequency, path_to_save_model, path_to_save_loss[i], path_to_save_predictions[i], device, training_length)
            model_dic = {
            "RNN_3": RNN_3().double().to(device),
            "GRU_3": GRU_3().double().to(device),
            "LSTM_3": LSTM_3().double().to(device),
            "RNN_attn": RNN_attn().double().to(device),
            "GRU_attn": GRU_attn().double().to(device),
            "LSTM_attn": LSTM_attn().double().to(device),
            "Transformer": Transformer().double().to(device),
            "GRU_1": GRU().double().to(device),
            "LSTM_1": model_LSTM_1().double().to(device),
            "RNN_1": RNN().double().to(device)
            }
            model_dic_keys = model_dic.keys()
            model_dic_keys_ls = list(model_dic_keys)
            best_model = f"{model_dic_keys_ls[model_num]}_best_train.pth"
            os.makedirs(os.path.dirname(path_to_save_predictions[database-1]), exist_ok=True)
            _rmse_1, r2_1, _rmse_2, r2_2, _rmse_3, r2_3, val_loss = inference(
                model_num, path_to_save_predictions[database-1], forecast_window,
                test_dataloader, device, path_model_exp,
                best_model, path_to_save_loss[database-1], current_exp,
                last_exp_num
                )
            train_loss = 0
            valid_loss = 0
            epoch_out = 0
            status = []
            status = [_rmse_1, r2_1, _rmse_2, r2_2, _rmse_3, r2_3, val_loss, train_loss, valid_loss, epoch_out]
            metric.append(status)
            print(f'Exp_num {j+1} has finished ({dataset_name})')
    
        # metric = pd.DataFrame(metric, columns = ['fc_1_rmse','fc_2_rmse','fc_3_rmse','fc_1_r2','fc_2_r2','fc_3_r2','test_loss','train_loss','best_model'],index = ['MLP_1','MLP_7','LSTM_1','LSTM_7','Transformer'])
        # metric = pd.DataFrame(metric, columns = ['fc_1_rmse','fc_1_r2','test_loss','train_loss','best_model'],index = ['MLP_1','MLP_7','LSTM_1','LSTM_7','Transformer'])

        metric_summary_fc1, metric_summary_fc2, metric_summary_fc3, model_dataset_name = stability_test(
            metric, Exp_num,
            result_loc, dataset_name
            )
        fc1_all_dataset.append(metric_summary_fc1)
        fc2_all_dataset.append(metric_summary_fc2)
        fc3_all_dataset.append(metric_summary_fc3)
        all_model_dataset_name.append(model_dataset_name)
    
    create_all_summary(fc1_all_dataset,fc2_all_dataset,fc3_all_dataset,all_model_dataset_name)
    # best_model = transformer(train_dataloader, epoch, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions[i], device)
