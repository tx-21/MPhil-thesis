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
    outdir = f'{result_loc}/metrics'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    full_path = os.path.join(outdir, outname)
    name_list.to_csv(full_path,index=True)
   

def stability_test(metric_all, Exp_num, result_loc, dataset_name):

        input_metric_number_fc1 = 5
        rmse_ls_1 = np.zeros([Exp_num])
        r2_ls_1 = np.zeros([Exp_num])
        test_loss_ls = np.zeros([Exp_num])
        train_loss_ls = np.zeros([Exp_num])
        valid_loss_ls = np.zeros([Exp_num])

        input_metric_number_fc2 = 2
        rmse_ls_2 = np.zeros([Exp_num])
        r2_ls_2 = np.zeros([Exp_num])

        input_metric_number_fc3 = 2
        rmse_ls_3 = np.zeros([Exp_num])
        r2_ls_3 = np.zeros([Exp_num])


        #for i in range(model_number):
        for j in range(Exp_num):
            # [exp_num, model, metric]
            rmse_ls_1[j] = metric_all[j][0] #model = i, exp_num = j
            r2_ls_1[j] = metric_all[j][1] #model = i, exp_num = j
            rmse_ls_2[j] = metric_all[j][2]
            r2_ls_2[j] = metric_all[j][3]
            rmse_ls_3[j] = metric_all[j][4] #model = i, exp_num = j
            r2_ls_3[j] = metric_all[j][5] #model = i, exp_num = j
            test_loss_ls[j] = metric_all[j][6]
            train_loss_ls[j] = metric_all[j][7]
            valid_loss_ls[j] = metric_all[j][8]
        
        metric_summary_fc1 = np.empty([input_metric_number_fc1*4])
        metric_summary_fc2 = np.empty([input_metric_number_fc2*4])
        metric_summary_fc3 = np.empty([input_metric_number_fc3*4])
        #3 models in total
        #std, min, max, mean of rmse, r2, test, train loss

        # or i in range(model_number):
        metric_summary_fc1[0] = round(rmse_ls_1.mean(),4)
        metric_summary_fc1[1] = round(rmse_ls_1.std(),4)
        metric_summary_fc1[2] = round(rmse_ls_1.max(),4)
        metric_summary_fc1[3] = round(rmse_ls_1.min(),4)
        metric_summary_fc1[4] = round(r2_ls_1.mean(),4)
        metric_summary_fc1[5] = round(r2_ls_1.std(),4)
        metric_summary_fc1[6] = round(r2_ls_1.max(),4)
        metric_summary_fc1[7] = round(r2_ls_1.min(),4)

        metric_summary_fc1[8] = round(test_loss_ls.mean(),4)
        metric_summary_fc1[9] = round(test_loss_ls.std(),4)
        metric_summary_fc1[10] = round(test_loss_ls.max(),4)
        metric_summary_fc1[11] = round(test_loss_ls.min(),4)

        metric_summary_fc1[12] = round(train_loss_ls.mean(),4)
        metric_summary_fc1[13] = round(train_loss_ls.std(),4)
        metric_summary_fc1[14] = round(train_loss_ls.max(),4)
        metric_summary_fc1[15] = round(train_loss_ls.min(),4)

        metric_summary_fc1[16] = round(valid_loss_ls.mean(),4)
        metric_summary_fc1[17] = round(valid_loss_ls.std(),4)
        metric_summary_fc1[18] = round(valid_loss_ls.max(),4)
        metric_summary_fc1[19] = round(valid_loss_ls.min(),4)
        # for i in range(model_number):
        metric_summary_fc2[0] = round(rmse_ls_2.mean(),4)
        metric_summary_fc2[1] = round(rmse_ls_2.std(),4)
        metric_summary_fc2[2] = round(rmse_ls_2.max(),4)
        metric_summary_fc2[3] = round(rmse_ls_2.min(),4)
        metric_summary_fc2[4] = round(r2_ls_2.mean(),4)
        metric_summary_fc2[5] = round(r2_ls_2.std(),4)
        metric_summary_fc2[6] = round(r2_ls_2.max(),4)
        metric_summary_fc2[7] = round(r2_ls_2.min(),4)
        
        # for i in range(model_number):
        metric_summary_fc3[0] = round(rmse_ls_3.mean(),4)
        metric_summary_fc3[1] = round(rmse_ls_3.std(),4)
        metric_summary_fc3[2] = round(rmse_ls_3.max(),4)
        metric_summary_fc3[3] = round(rmse_ls_3.min(),4)
        metric_summary_fc3[4] = round(r2_ls_3.mean(),4)
        metric_summary_fc3[5] = round(r2_ls_3.std(),4)
        metric_summary_fc3[6] = round(r2_ls_3.max(),4)
        metric_summary_fc3[7] = round(r2_ls_3.min(),4)

        index_model_name = [f'{dataset_name}']
        metric_name_1 = ['fc1_rmse_mean','fc1_rmse_std','fc1_rmse_max','fc1_rmse_min','fc1_r2_mean','fc1_r2_std','fc1_r2_max','fc1_r2_min']
        metric_name_2 = ['fc2_rmse_mean','fc2_rmse_std','fc2_rmse_max','fc2_rmse_min','fc2_r2_mean','fc2_r2_std','fc2_r2_max','fc2_r2_min']
        metric_name_3 = ['fc3_rmse_mean','fc3_rmse_std','fc3_rmse_max','fc3_rmse_min','fc3_r2_mean','fc3_r2_std','fc3_r2_max','fc3_r2_min']

        metric_names = metric_name_1 + ['test_loss_mean','test_loss_std','test_loss_max','test_loss_min','train_loss_mean','train_loss_std','train_loss_max','train_loss_min','valid_loss_mean','valid_loss_std','valid_loss_max','valid_loss_min' ]
        # print(np.shape(rmse_ls_1))
        # print(rmse_ls_1)
        # print(np.shape(rmse_ls_1.ravel()))
        # print(rmse_ls_1)
        # exit()
        rmse_ls_1 = pd.DataFrame(rmse_ls_1.flatten(), columns = index_model_name).T
        r2_ls_1 = pd.DataFrame(r2_ls_1, columns = index_model_name).T
        rmse_ls_2 = pd.DataFrame(rmse_ls_2, columns = index_model_name).T
        r2_ls_2 = pd.DataFrame(r2_ls_2, columns = index_model_name).T
        rmse_ls_3 = pd.DataFrame(rmse_ls_3, columns = index_model_name).T
        r2_ls_3 = pd.DataFrame(r2_ls_3, columns = index_model_name).T        
        test_loss_ls = pd.DataFrame(test_loss_ls, columns = index_model_name).T
        train_loss_ls = pd.DataFrame(train_loss_ls, columns = index_model_name).T
        valid_loss_ls = pd.DataFrame(valid_loss_ls, columns = index_model_name).T
        metric_summary_fc1 = pd.DataFrame(metric_summary_fc1, index = metric_names,  columns = index_model_name).T
        metric_summary_fc2 = pd.DataFrame(metric_summary_fc2, index = metric_name_2, columns = index_model_name).T
        metric_summary_fc3 = pd.DataFrame(metric_summary_fc3, index = metric_name_3, columns = index_model_name).T
        
        create_csv(rmse_ls_1,'rmse_1.csv',result_loc)
        create_csv(r2_ls_1,'r2_1.csv',result_loc)
        create_csv(rmse_ls_2,'rmse_2.csv',result_loc)
        create_csv(r2_ls_2,'r2_2.csv',result_loc)
        create_csv(rmse_ls_3,'rmse_3.csv',result_loc)
        create_csv(r2_ls_3,'r2_3.csv',result_loc)
        create_csv(test_loss_ls,'test_loss.csv',result_loc)
        create_csv(train_loss_ls,'train_loss.csv',result_loc)
        create_csv(valid_loss_ls,'valid_loss.csv',result_loc)
        create_csv(metric_summary_fc1,'summary_metric_fc1.csv',result_loc)
        create_csv(metric_summary_fc2,'summary_metric_fc2.csv',result_loc)
        create_csv(metric_summary_fc3,'summary_metric_fc3.csv',result_loc)

        return metric_summary_fc1,metric_summary_fc2,metric_summary_fc3,index_model_name

def create_all_summary(fc1_ls,fc2_ls,fc3_ls,all_model_dataset_name):

    index_name = np.array(all_model_dataset_name).flatten()
    metric_name_1 = ['fc1_rmse_mean','fc1_rmse_std','fc1_rmse_max','fc1_rmse_min','fc1_r2_mean','fc1_r2_std','fc1_r2_max','fc1_r2_min']
    metric_name_2 = ['fc2_rmse_mean','fc2_rmse_std','fc2_rmse_max','fc2_rmse_min','fc2_r2_mean','fc2_r2_std','fc2_r2_max','fc2_r2_min']
    metric_name_3 = ['fc3_rmse_mean','fc3_rmse_std','fc3_rmse_max','fc3_rmse_min','fc3_r2_mean','fc3_r2_std','fc3_r2_max','fc3_r2_min']
    metric_names = metric_name_1 + ['test_loss_mean','test_loss_std','test_loss_max','test_loss_min','train_loss_mean','train_loss_std','train_loss_max','train_loss_min','valid_loss_mean','valid_loss_std','valid_loss_max','valid_loss_min']

    fc1_all_df = pd.DataFrame(np.array(fc1_ls).reshape(-1,20),columns = metric_names, index = index_name)
    fc2_all_df = pd.DataFrame(np.array(fc2_ls).reshape(-1,8),columns = metric_name_2, index = index_name)
    fc3_all_df = pd.DataFrame(np.array(fc3_ls).reshape(-1,8),columns = metric_name_3, index = index_name)

    fc1_all_df.to_csv('results/fc1_all_dataset.csv',index=True)
    fc2_all_df.to_csv('results/fc2_all_dataset.csv',index=True)
    fc3_all_df.to_csv('results/fc3_all_dataset.csv',index=True)


# Remove all files from previous executions and re-run the model.
def clean_directory():

    if os.path.exists('results'):
        shutil.rmtree('results')
    # if os.path.exists('save_loss_RNN'):
    #     shutil.rmtree('save_loss_RNN')
    # if os.path.exists('save_loss_GRU'):
    #     shutil.rmtree('save_loss_GRU')
    # if os.path.exists('save_loss_DNN'):
    #     shutil.rmtree('save_loss_DNN')
    # if os.path.exists('save_loss_LSTM'):
    #     shutil.rmtree('save_loss_LSTM')
    # if os.path.exists('save_model'): 
    #     shutil.rmtree('save_model')
    # if os.path.exists('save_predictions_CNN'): 
    #     shutil.rmtree('save_predictions_CNN')
    # if os.path.exists('save_predictions_RNN'): 
    #     shutil.rmtree('save_predictions_RNN')
    # if os.path.exists('save_predictions_GRU'): 
    #     shutil.rmtree('save_predictions_GRU')
    # if os.path.exists('save_predictions_DNN'): 
    #     shutil.rmtree('save_predictions_DNN')
    # if os.path.exists('save_predictions_LSTM'): 
    #     shutil.rmtree('save_predictions_LSTM')

   # os.mkdir("save_loss_CNN")
   # os.mkdir("save_loss_RNN")
   # os.mkdir("save_loss_GRU")
   # os.mkdir("save_loss_DNN")
   # os.mkdir("save_loss_LSTM")
   # os.mkdir("save_model")
   # os.mkdir("save_predictions_CNN")
   # os.mkdir("save_predictions_RNN")
   # os.mkdir("save_predictions_GRU")
   # os.mkdir("save_predictions_DNN")
   # os.mkdir("save_predictions_LSTM")