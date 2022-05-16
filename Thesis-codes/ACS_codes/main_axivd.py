import argparse
from train import *
from DataLoader import *
from DataLoader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from helpers import *
from inference import *
import warnings
warnings.simplefilter("ignore", UserWarning)

def main(
    epoch: int = 400,
    k: int = 60,
    batch_size: int = 1,
    frequency: int = 100,
    training_length = 24,
    forecast_window = 1,
    lr = 0.0003,
    train_csv = "train_dataset.csv",
    test_csv = "test_dataset.csv",
    path_to_save_model = "save_model/",
    path_to_save_loss_1 = "save_loss_1/",
    path_to_save_loss_2 = "save_loss_2/",
    path_to_save_loss_3 = "save_loss_3/",
    path_to_save_loss_4 = "save_loss_4/",
    path_to_save_loss_5 = "save_loss_5/",
    path_to_save_predictions_1 = "save_predictions_1/",
    path_to_save_predictions_2 = "save_predictions_2/",
    path_to_save_predictions_3 = "save_predictions_3/",
    path_to_save_predictions_4 = "save_predictions_4/",
    path_to_save_predictions_5 = "save_predictions_5/",
    device = 'cpu'
):

    clean_directory()
    
    train_dataset = TrainDataset(csv_name = train_csv, root_dir = "Data/", training_length = training_length, forecast_window = forecast_window)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = TestDataset(csv_name = test_csv, root_dir = "Data/", training_length = training_length, forecast_window = forecast_window)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #k was removed for using train_teacher_forcing
    model_number = 5
    #num1 = MLP_1, num2 = MLP_7, num3 = LSTM_1, num4 = LSTM_7, num5 = Transformer
    load_best_model = []
    path_to_save_predictions = [path_to_save_predictions_1,path_to_save_predictions_2,path_to_save_predictions_3,path_to_save_predictions_4,path_to_save_predictions_5]
    path_to_save_loss = [path_to_save_loss_1,path_to_save_loss_2,path_to_save_loss_3,path_to_save_loss_4,path_to_save_loss_5]
    metric = []
    for i in range(model_number):
        load_best_model.append(teacher_forcing(i, train_dataloader, epoch, lr, k, frequency, path_to_save_model, path_to_save_loss[i], path_to_save_predictions[i], device))
        # load_best_model.append(scheduled_sampling(i, train_dataloader, epoch, lr, k, frequency, path_to_save_model, path_to_save_loss[i], path_to_save_predictions[i], device, training_length))
        metric.append(inference(i, path_to_save_predictions[i], forecast_window, test_dataloader, device, path_to_save_model, load_best_model[i][1],path_to_save_loss[i]))
        metric[i].extend(load_best_model[i])
    
    # metric = pd.DataFrame(metric, columns = ['fc_1_rmse','fc_2_rmse','fc_3_rmse','fc_1_r2','fc_2_r2','fc_3_r2','test_loss','train_loss','best_model'],index = ['MLP_1','MLP_7','LSTM_1','LSTM_7','Transformer'])
    metric = pd.DataFrame(metric, columns = ['fc_1_rmse','fc_1_r2','test_loss','train_loss','best_model'],index = ['MLP_1','MLP_7','LSTM_1','LSTM_7','Transformer'])
    metric.to_csv('metrics.csv')    
    # best_model = transformer(train_dataloader, epoch, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions[i], device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--path_to_save_model",type=str,default="save_model/")
    parser.add_argument("--path_to_save_loss_1",type=str,default="save_loss_1/")
    parser.add_argument("--path_to_save_loss_2",type=str,default="save_loss_2/")
    parser.add_argument("--path_to_save_loss_3",type=str,default="save_loss_3/")
    parser.add_argument("--path_to_save_loss_4",type=str,default="save_loss_4/")
    parser.add_argument("--path_to_save_loss_5",type=str,default="save_loss_5/")
    parser.add_argument("--path_to_save_predictions_1",type=str,default="save_predictions_1/")
    parser.add_argument("--path_to_save_predictions_2",type=str,default="save_predictions_2/")
    parser.add_argument("--path_to_save_predictions_3",type=str,default="save_predictions_3/")
    parser.add_argument("--path_to_save_predictions_4",type=str,default="save_predictions_4/")
    parser.add_argument("--path_to_save_predictions_5",type=str,default="save_predictions_5/")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        epoch=args.epoch,
        k = args.k,
        batch_size=args.batch_size,
        lr=args.lr,
        frequency=args.frequency,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss_1=args.path_to_save_loss_1,
        path_to_save_loss_2=args.path_to_save_loss_2,
        path_to_save_loss_3=args.path_to_save_loss_3,
        path_to_save_loss_4=args.path_to_save_loss_4,
        path_to_save_loss_5=args.path_to_save_loss_5,
        path_to_save_predictions_1=args.path_to_save_predictions_1,
        path_to_save_predictions_2=args.path_to_save_predictions_2,
        path_to_save_predictions_3=args.path_to_save_predictions_3,
        path_to_save_predictions_4=args.path_to_save_predictions_4,
        path_to_save_predictions_5=args.path_to_save_predictions_5,
        device=args.device,
    )

