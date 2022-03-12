import os, shutil
import numpy as np
import pandas as pd
# save train or validation loss
def log_loss(loss_val : float, path_to_save_loss : str, train : bool = True):
    if train:
        file_name = "train_loss.txt"
    else:
        file_name = "val_loss.txt"

    path_to_file = path_to_save_loss+file_name
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "a") as f:
        f.write(str(loss_val)+"\n")
        f.close()

def log_test_loss(loss_val : float, path_to_save_loss : str):

    file_name = "test_loss.txt"

    path_to_file = path_to_save_loss+file_name
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "a") as f:
        f.write(str(loss_val)+"\n")
        f.close()

# Exponential Moving Average, https://en.wikipedia.org/wiki/Moving_average
def EMA(values, alpha=0.1):
    ema_values = [values[0]]
    for idx, item in enumerate(values[1:]):
        ema_values.append(alpha*item + (1-alpha)*ema_values[idx])
    return ema_values

def stability_test(metric_all, model_number, Exp_num):

        input_metric_number = 4
        r2_ls = np.zeros([model_number,Exp_num])
        rmse_ls = np.zeros([model_number,Exp_num])
        test_loss_ls = np.zeros([model_number,Exp_num])
        train_loss_ls = np.zeros([model_number,Exp_num])

        for i in range(model_number):
            for j in range(Exp_num):
                # [exp_num, model, metric]

                rmse_ls[i,j] = metric_all[j][i][0] #model = i, exp_num = j
                r2_ls[i,j] = metric_all[j][i][1] #model = i, exp_num = j
                test_loss_ls[i,j] = metric_all[j][i][2]
                train_loss_ls[i,j] = metric_all[j][i][3]
        
        metric_summary = np.empty([5, input_metric_number*4])
        #std, min, max, mean of rmse, r2, test, train loss

        for i in range(model_number):
            metric_summary[i,0] = round(rmse_ls[i].mean(),4)
            metric_summary[i,1] = round(rmse_ls[i].std(),4)
            metric_summary[i,2] = round(rmse_ls[i].max(),4)
            metric_summary[i,3] = round(rmse_ls[i].min(),4)
            metric_summary[i,4] = round(r2_ls[i].mean(),4)
            metric_summary[i,5] = round(r2_ls[i].std(),4)
            metric_summary[i,6] = round(r2_ls[i].max(),4)
            metric_summary[i,7] = round(r2_ls[i].min(),4)
            metric_summary[i,8] = round(test_loss_ls[i].mean(),4)
            metric_summary[i,9] = round(test_loss_ls[i].std(),4)
            metric_summary[i,10] = round(test_loss_ls[i].max(),4)
            metric_summary[i,11] = round(test_loss_ls[i].min(),4)
            metric_summary[i,12] = round(train_loss_ls[i].mean(),4)
            metric_summary[i,13] = round(train_loss_ls[i].std(),4)
            metric_summary[i,14] = round(train_loss_ls[i].max(),4)
            metric_summary[i,15] = round(train_loss_ls[i].min(),4)

        metric_names = ['rmse_mean','rmse_std','rmse_max','rmse_min','r2_mean','r2_std','r2_max','r2_min','test_loss_mean','test_loss_std','test_loss_max','test_loss_min','train_loss_mean','train_loss_std','train_loss_max','train_loss_min']
        rmse_ls = pd.DataFrame(rmse_ls, index = ['MLP_1','MLP_7','LSTM_1','LSTM_7','Transformer'])
        r2_ls = pd.DataFrame(r2_ls, index = ['MLP_1','MLP_7','LSTM_1','LSTM_7','Transformer'])
        test_loss_ls = pd.DataFrame(test_loss_ls, index = ['MLP_1','MLP_7','LSTM_1','LSTM_7','Transformer'])
        train_loss_ls = pd.DataFrame(train_loss_ls, index = ['MLP_1','MLP_7','LSTM_1','LSTM_7','Transformer'])
        metric_summary = pd.DataFrame(metric_summary, columns = metric_names, index = ['MLP_1','MLP_7','LSTM_1','LSTM_7','Transformer'])
        
        rmse_ls.to_csv('Result_metrics/rmse.csv')
        r2_ls.to_csv('Result_metrics/r2.csv')
        test_loss_ls.to_csv('Result_metrics/test_loss.csv')
        train_loss_ls.to_csv('Result_metrics/train_loss.csv')
        metric_summary.to_csv('Result_metrics/summary_metric.csv')
        
        
        return metric_summary

# Remove all files from previous executions and re-run the model.
def clean_directory():

    if os.path.exists('save_loss_1'):
        shutil.rmtree('save_loss_1')
    if os.path.exists('save_loss_2'):
        shutil.rmtree('save_loss_2')
    if os.path.exists('save_loss_3'):
        shutil.rmtree('save_loss_3')
    if os.path.exists('save_loss_4'):
        shutil.rmtree('save_loss_4')
    if os.path.exists('save_loss_5'):
        shutil.rmtree('save_loss_5')
    if os.path.exists('save_model'): 
        shutil.rmtree('save_model')
    if os.path.exists('save_predictions_1'): 
        shutil.rmtree('save_predictions_1')
    if os.path.exists('save_predictions_2'): 
        shutil.rmtree('save_predictions_2')
    if os.path.exists('save_predictions_3'): 
        shutil.rmtree('save_predictions_3')
    if os.path.exists('save_predictions_4'): 
        shutil.rmtree('save_predictions_4')
    if os.path.exists('save_predictions_5'): 
        shutil.rmtree('save_predictions_5')
    
    os.mkdir("save_loss_1")
    os.mkdir("save_loss_2")
    os.mkdir("save_loss_3")
    os.mkdir("save_loss_4")
    os.mkdir("save_loss_5")
    os.mkdir("save_model")
    os.mkdir("save_predictions_1")
    os.mkdir("save_predictions_2")
    os.mkdir("save_predictions_3")
    os.mkdir("save_predictions_4")
    os.mkdir("save_predictions_5")