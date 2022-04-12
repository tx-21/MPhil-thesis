Exp-1: Looking for best dataset+method combination
Exp-1.1: Use RF+datasets to see the performance.
Exp-2: Use several best combinations to test with positional encoding
Exp-2.1: Use Transformer+datasets
Exp-2.2: Use RF + positional encoding


# Testing
* Terminal Exp 1
  * Testing sheduler. (compare with and without scheduler at lr=1e-04, epoch=50)

# Next
## 4/11
* [x] To find the optimal lr for exp 1.
  * [x] locate the good range of lr for exp1
* [ ] proceed to exp2, which is to use positional encoding to improve on the already good model from exp1. (need to add gru, lstm, rnn with attn)
  * [x] find out what's the problem with transformer, or just simply give up on it.
* [x] also test positional encoding on RF.
  * No effect
## 4/12
# Results
* RF can only predict NH$_{3}$N wtih 0.34 RMSE (fc1).
* lr=1e-05 requires 150-200 epoch to reach optimal.
* if train loss < 2, it will overfit
* some combination of lr and epoch have been tried on transformer model and the results are poorer than using other deep learning models
* Using positional encoding on RF didn't improve the model performance
## epoch
* for lstm-sg7, 
# ideas


# To be solved
* [x] Optimal training epoch?
  * for lr=1e-05, epoch should be around 200
* [ ] Can transformer or attn can really improve the model performance?
* [x] Change to lr to lr=5*1e-05 in exp2 and exp1
* [ ] attn seems to improve on the std?
* [x] transformer seems not working.
  * [x] transformer has poor performance in univariate forecasting in our case
performance.
# Exp results
## Exp 1
* nh3 predict nh3 using cnn, rnn, gru, dnn, lstm with 8 datasets.
  * result-3, epoch = 20, Exp_num=10, RNN-ew2	0.5277	0.0787	27.6076 # rmse, test, train
  * result-4, epoch = 50, Exp_num=3, RNN-ew2	0.3193	0.0538	11.0396
  * result-5, epoch = 100, Exp_num=3, RNN-ew2	0.2635	0.0465	8.0311
  * result-6, epoch = 150, Exp_num=5, RNN-ew3	0.2388	0.0436	5.5158
  * result-7, epoch = 100, Exp_num=3, lr = 3*1e-04
  * result-8, epoch = 50, Exp_num=5, lr = 5*1e-05
  * result-9, epoch = 50, Exp_num=3, lr = 1e-04 
 
## Exp 1.1
* Use RF to predict nh3 with 9 datasets.
  * result-4, estimator=100, Exp_num=10, sg5	RF	0.3364 0.8976 0.2337		
  * result-5, estimator=500, Exp_num=10, sg7	RF	0.3387 0.8858 0.2022
> RF prediction accuracy is only around 0.34 and stop decreasing with the increase of estimator.

## Exp 2
* Use RNN, DNN, RNN-attn with the corresponding data preprocessing method to predict nh3. (with positional encoding) and also add transformer (with 1 dataset). 
  * results-1, epoch = 50, Exp_num=5, RNN-ews	0.4318	0.0627	14.426
  * results-2, epoch = 20, Exp_num=5, RNN-ews	0.6202	0.0844	26.7278
  * results-3, epoch = 100, Exp_num=5, RNN-ews	0.2834	0.0471	6.9025  
  * results-4, epoch = 150, Exp_num=5, RNN-ews	0.2811	0.0456	6.214
  * results-5, epoch = 200, Exp_num=5, RNN-ews	0.2626	0.0445	5.168
  * results-6, epoch = 100, Exp_num=5, lr=4*1e-05, RNN-ews	0.2445	0.0418	4.7295


## Exp 2.1
* [ ] If transformer is not good, then focus on the attn of lstm/rnn/gru. 
* [ ] try different hyperparameter
* Use Transformer to predict (with positional encoding)
  * result-1, epoch = 20, Exp_num=5, lr = 1e-05, tf-ews	0.7042	0.1022	11.7339
  * result-2, epoch = 50, Exp_num=5, lr = 1e-05, tf-ews	0.7141	0.1205	7.4256
  * result-3, epoch = 10, Exp_num=5, lr = 1e-05, tf-ews	0.7112	0.0951	25.0694
  * result-4, epoch = 100, Exp_num=5, lr = 1e-05, tf-ews	0.7724	0.1243	5.2446
  * result-5, epoch = 100, Exp_num=5, lr = 1e-05, tf_model_layer=3, tf-ews	0.6875	0.1255	3.5095
  * result-6, epoch = 50, Exp_num=1, lr = 1e-04,  tf_model_layer=3, 

## Exp 4-val
* no scheduler, lr=5*1e-05, epoch=50
  * RNN-ews 0.2707	0.0456	4.386	4.386
* with scheduler
  * lr=1e-04, epoch=50,  factor=0.5, patience=2  >> RNN-ews	0.2312	0.0426	4.3966	1.239 # rmse, test, train, valid
* no scheduler
  * lr=1e-04, epoch=50,  factor=0.5, patience=2  >> RNN-ews	0.2422	0.0441	3.8699	1.3102




# file saving directory
* Exp-1
  * data
    * test
    * train
      * 8 datasets
  * results
    * fc-1-summary (fc1+test/loss)
    * fc-2-summary (fc2)
    * fc-3-summary (fc3)
    * obs
      * metrics
      * save_model
      * loss_models*5
      * pred_models*5
    * etc... (7 datasets)
* Exp-2
* Exp-3
# raw-data-pre-processing.ipynb
### Input
* Raw NH$_{3}$N and colour data from 12/23 to 1/22.
---
## Exp-1
### Purpose
* Perform data pre-processing on NH3.
*  Perform outlier removal on NH3.
*  Generate pre-processed dataset for all four Exp-.
* Demonstate/explain the reasons behind selecting SG
and EWMA filters and OR by visulization.
### Tasks
* [x] Apply SG filters (windows = 5,7,9) to NH3.
* [x] Apply EWMA filters (span = 2,3,4) to NH3.
* [x] Apply outlier removal to NH3.
* [x] Plot SG filters.
* [x] Plot EWMA filters.
* [x] Plot peak analysis.
* [x] use makedirs to create folders
### Output
1. train.csv of obs, sg5, sg7, sg9, ew2, ew3, ew4, and or.
2. test.csv
---
## Exp-2
### Purpose
### Task
### Output
---
## Exp-3
### Purpose
### Task
### Output
---
# Preprocessing.py
## Exp-1
### Input
* train.csv of obs, sg5, sg7, sg9, ew2, ew3, ew4, and or.
* test.csv.
### Purpose
* Add positional encoding to the input train/test data.
### Task
* [x] Add hour and day positional encoding to NH$_{3}$N data.
* [x] Create only the hour and day positional encoding
* [x] Export to the right directories
### Output
* train_dataset.csv of obs, sg5, sg7, sg9, ew2, ew3, ew4, and or.
* test_dataset.csv.
---
## Exp-2
---
## Exp-3

# models.py
## Exp-1
### Purpose
* Generate available models for training.
### Tasks
* Writing model
  * [ ] Random forest
  * [x] CNN
    * src has been changed into [: ,: ,0 ] (1 feature)
    * In CNN, the src is required to be premuted ([N ,: ,features instead of [: ,N ,features])
  * [x] RNN
  * [x] GRU
  * [x] DNN
  * [x] LSTM
* [x] Check the input data into the models follows the right format stated by pytorch website.
* [x] Check the input, output size of the model.
---
## Exp-2
* [ ] DNN-5
* [ ] LSTM-5
* [ ] Attn-LSTM-5
* [ ] Transformer
---
## Exp-3
 
# Dataloader.py
### Purpose
* Input the train and test dataset one by one.
---
## TrainDataset
* [x] check the len__
* [x] the input and target should only input with positional encoding of hour and day
## TestDataset
* [x] The len needs to be change according to the length of the testdataset.
# train.py
## teacher-forcing func
### Purpose
* To train time series models with teacher forcing.
* To save train loss to designated location
* To plot the train loss (in the case of exp 1, it's not neccessary)
### Tasks
* [x] Create a dictionary for model selection.
* [x] Determine showing the loss in n epoch
* [x] Decide how to calculate the train loss.
  * The train loss is the MSE for 23 hours.
## scheduled sampling
### Purpose
* Same as teacher forcing but changed to scheduled sampling.
# Inference.py
### Purpose
* Input the test dataset in trained model to forecast.
### Tasks
* [x] Decide the forecast horizon.
* [x] Decide how to calculate the test loss
  * test loss = (test loss /= len(dataloader))

# helpers.py
### Purpose
* Write loss file save location
* Other functions
### Task
#### Directory related
##### Modify the file saving path of log_loss
  * log_loss/helpers.py > teacher_forcing/train.py > main/main_load.py > path_to_save_loss[i]
    > results/obs/loss/path_to_save_loss[i]
##### log_test_loss
  * log_test_loss/helprs.py > inference/inference.py > path_to_save_loss[i]
    > results/obs/loss/path_to_save_loss[i]
##### stability_test
  * create_csv > stability_test/helpers.py > stability_test/main_load.py > result_loc
    > results/obs/metircs/{}.csv
##### clean directory function
#### Compile all the results from 8 dataset


# main-load.py
### purpose
* To input all the required parameters for main.py to run.
* Organize all the script and output directories of results.
### Task
* [ ] The ways to input different databases.
* [ ] loss saving path
* [ ] pred saving path
  * plot_training/plot.py > plot_training/train.py > teacher_forcing, inference/main_load.py > path_to_save_predictions[i]
    > results/obs/pred/save_predictions[i]
* [ ] model saving path
    > results/obs/save_model
* [ ] hyperparameters
### Constrains
Since the models are run for n exp times, the plotting will only be executed in the last Exp. 

# main.py
### Purpose
* Input arguments and run with python.

# plot.py
## Exp-1
### Tasks
#### plot_prediction_horizon (link to inference.py)
* save to __path_to_save_predictions__
* [x] Change the unit and range in y axis
* [x] Change the date_range
#### plot_prediction (link to inference.py)
* save to __path_to_save_predictions__
#### plot_loss (link to train.py and inference.py)
* save to __path_to_save_loss__
#### plot_training(link to train.py)
* save to __path_to_save_predictions__
