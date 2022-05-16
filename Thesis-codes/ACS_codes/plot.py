import matplotlib.pyplot as plt
from helpers import EMA
from icecream import ic 
import numpy as np
import seaborn as sns
import torch
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def rmse(pred, actual):
    error = np.subtract(pred, actual)
    sqerror= np.sum(np.square(error))/actual.shape[0]
    return np.sqrt(sqerror)

def plot_prediction_horizon(pred_fh, true_fh, horizon_num, Mname, path_to_save,date_range):

    # %config InlineBackend.figure_format='retina'
    params = {'mathtext.default': 'regular','font.size': 10}          
    plt.rcParams.update(params)
    sns.set_style("ticks")

    R2 = r2_score(true_fh, pred_fh) #[y_true,y_pred]
    rmse_val = rmse(pred_fh, true_fh) #[y_pred,y_true]

    fig, ax = plt.subplots(figsize=(5,4))
    plt.suptitle(f'Model:{Mname}\nObserved vs Predicted (RMSE:{round(rmse_val,4)}, R2:{round(R2,4)})')

    size=5
    # plot pred
    plt.plot(np.arange(0,len(pred_fh)), pred_fh.values, '-', color = 'red', lw=1.2, label='prediction $NH_3-N$')
    plt.scatter(np.arange(0,len(pred_fh)), pred_fh.values, s=size, color='darkred')

    # plot true
    ax.scatter(np.arange(0,len(pred_fh)), true_fh.values,s=size,color='blue')
    ax.plot(np.arange(0,len(pred_fh)), true_fh.values, label='Observed $NH_3-N$',lw=1.2)

    ax.legend(loc='upper right')
    #ax.grid(which='both')
    ax.set_xlabel('Date')
    ax.set_ylabel('mg/L')
    ax.margins(x=0.01)

    plt.grid(b=True, which='major', linestyle = 'solid')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle = 'dashed', alpha=0.5)
    xmajorLocator = MultipleLocator(24)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.grid(which='major', ls='-')
    #11/20~12/18
    ax.set_xticklabels(date_range)
    plt.tight_layout()
    plt.savefig(path_to_save+f"{Mname}_pred_Step{horizon_num}.png",dpi=300,bbox_inches='tight') 
    plt.close()
                    
    return round(rmse_val,4), round(R2,4)

def plot_loss(model_name, path_to_save, train=True):
    plt.rcParams.update({'font.size': 10}) #appoint the fontsize in all the plots
    with open(path_to_save + "/train_loss.txt", 'r') as f:
        loss_list = [float(line) for line in f.readlines()]
    if train:
        title = "Train"
    else:
        title = "Validation"
    EMA_loss = EMA(loss_list)
    plt.plot(loss_list, label = "loss")
    plt.plot(EMA_loss, label="EMA loss") #plot trend line with exponential average curve
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title+"_loss")
    plt.savefig(path_to_save+f"/{title}_{model_name}.png")
    plt.close()


def plot_prediction(title, path_to_save, src, tgt, prediction, index_in, index_tar):

    idx_scr = index_in[0, 1:].tolist()
    idx_tgt = index_tar[0].tolist()
    # print(idx_scr)
    # print(idx_tgt)
    # exit()
    idx_pred = [i for i in range(idx_scr[0] + 1, idx_tgt[-1] + 1)] #t2 - t61

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 16})

    # connect with last elemenet in src
    # tgt = np.append(src[-1], tgt.flatten())
    # prediction = np.append(src[-1], prediction.flatten())

    # plotting
    plt.plot(idx_scr, src, '-', color = 'blue', label = 'Input', linewidth=2)
    plt.plot(idx_tgt, tgt, '-', color = 'indigo', label = 'Target', linewidth=2)
    plt.plot(idx_pred, prediction,'--', color = 'limegreen', label = 'Forecast', linewidth=2)

    #formatting
    plt.grid(b=True, which='major', linestyle = 'solid')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle = 'dashed', alpha=0.5)
    plt.xlabel("Time Elapsed")
    plt.ylabel("NH3-N (mg/L)")
    plt.legend()
    plt.title("Forecast from ....")

    # save
    plt.savefig(path_to_save+f"Prediction_{title}.png")
    plt.close()

def plot_training(epoch, path_to_save, src, prediction, index_in, index_tar, model_name):

    # idx_scr = index_in.tolist()[0]
    # idx_tar = index_tar.tolist()[0]
    # idx_pred = idx_scr.append(idx_tar.append([idx_tar[-1] + 1]))

    idx_scr = [i for i in range(len(src))]
    idx_pred = [i for i in range(1, len(prediction)+1)]

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 18})
    plt.grid(b=True, which='major', linestyle = '-')
    plt.grid(b=True, which='minor', linestyle = '--', alpha=0.5)
    plt.minorticks_on()

    plt.plot(idx_scr, src, 'o-.', color = 'blue', label = 'input sequence', linewidth=1)
    plt.plot(idx_pred, prediction, 'o-.', color = 'limegreen', label = 'prediction sequence', linewidth=1)

    plt.title(f"Teaching Forcing from {model_name}" + ", Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("NH3-N (mg/L)")
    plt.legend()
    plt.savefig(path_to_save+f"/{model_name}_Epoch_{str(epoch)}.png")
    plt.close()

def plot_training_3(epoch, path_to_save, src, sampled_src, prediction, index_in, index_tar):

    # idx_scr = index_in.tolist()[0]
    # idx_tar = index_tar.tolist()[0]
    # idx_pred = idx_scr.append(idx_tar.append([idx_tar[-1] + 1]))

    idx_scr = [i for i in range(len(src))]
    idx_pred = [i for i in range(1, len(prediction)+1)]
    idx_sampled_src = [i for i in range(len(sampled_src))]

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 18})
    plt.grid(b=True, which='major', linestyle = '-')
    plt.grid(b=True, which='minor', linestyle = '--', alpha=0.5)
    plt.minorticks_on()

    ## REMOVE DROPOUT FOR THIS PLOT TO APPEAR AS EXPECTED !! DROPOUT INTERFERES WITH HOW THE SAMPLED SOURCES ARE PLOTTED
    plt.plot(idx_sampled_src, sampled_src, 'o-.', color='red', label = 'sampled source', linewidth=1, markersize=10)
    plt.plot(idx_scr, src, 'o-.', color = 'blue', label = 'input sequence', linewidth=1)
    plt.plot(idx_pred, prediction, 'o-.', color = 'limegreen', label = 'prediction sequence', linewidth=1)
    plt.title("Teaching Forcing from ...... " + ", Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("NH3-N (mg/L)")
    plt.legend()
    plt.savefig(path_to_save+f"/Epoch_{str(epoch)}.png")
    plt.close()