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
