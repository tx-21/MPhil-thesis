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

def create_csv(name_list,outname,result_loc):
    outname = outname
    outdir = f'metrics/{result_loc}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    full_path = os.path.join(outdir, outname)
    name_list.to_csv(full_path,index=False)
   

def stability_test(metric_all, model_number, Exp_num, result_loc):

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
            
        index_model_name = ['CNN','RNN','GRU','DNN','LSTM']
        metric_names = ['rmse_mean','rmse_std','rmse_max','rmse_min','r2_mean','r2_std','r2_max','r2_min','test_loss_mean','test_loss_std','test_loss_max','test_loss_min','train_loss_mean','train_loss_std','train_loss_max','train_loss_min']
        rmse_ls = pd.DataFrame(rmse_ls, index = index_model_name)
        r2_ls = pd.DataFrame(r2_ls, index = index_model_name)
        test_loss_ls = pd.DataFrame(test_loss_ls, index = index_model_name)
        train_loss_ls = pd.DataFrame(train_loss_ls, index = index_model_name)
        metric_summary = pd.DataFrame(metric_summary, columns = metric_names, index = index_model_name)
        
        create_csv(rmse_ls,'rmse.csv',result_loc)
        create_csv(r2_ls,'r2.csv',result_loc)
        create_csv(test_loss_ls,'test_loss.csv',result_loc)
        create_csv(train_loss_ls,'train_loss.csv',result_loc)
        create_csv(metric_summary,'summary_metric.csv',result_loc)
        
        return metric_summary

# Remove all files from previous executions and re-run the model.
def clean_directory():

    if os.path.exists('save_loss_CNN'):
        shutil.rmtree('save_loss_CNN')
    if os.path.exists('save_loss_RNN'):
        shutil.rmtree('save_loss_RNN')
    if os.path.exists('save_loss_GRU'):
        shutil.rmtree('save_loss_GRU')
    if os.path.exists('save_loss_DNN'):
        shutil.rmtree('save_loss_DNN')
    if os.path.exists('save_loss_LSTM'):
        shutil.rmtree('save_loss_LSTM')
    if os.path.exists('save_model'): 
        shutil.rmtree('save_model')
    if os.path.exists('save_predictions_CNN'): 
        shutil.rmtree('save_predictions_CNN')
    if os.path.exists('save_predictions_RNN'): 
        shutil.rmtree('save_predictions_RNN')
    if os.path.exists('save_predictions_GRU'): 
        shutil.rmtree('save_predictions_GRU')
    if os.path.exists('save_predictions_DNN'): 
        shutil.rmtree('save_predictions_DNN')
    if os.path.exists('save_predictions_LSTM'): 
        shutil.rmtree('save_predictions_LSTM')
    
    os.mkdir("save_loss_CNN")
    os.mkdir("save_loss_RNN")
    os.mkdir("save_loss_GRU")
    os.mkdir("save_loss_DNN")
    os.mkdir("save_loss_LSTM")
    os.mkdir("save_model")
    os.mkdir("save_predictions_CNN")
    os.mkdir("save_predictions_RNN")
    os.mkdir("save_predictions_GRU")
    os.mkdir("save_predictions_DNN")
    os.mkdir("save_predictions_LSTM")