# raw-data-pre-processing.ipynb
### Input
* Raw NH$_{3}$N and colour data from 12/23 to 1/20.
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
* [ ] Plot SG filters.
* [ ] Plot EWMA filters.
* [ ] Plot peak analysis.
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
## Checklist
* [ ] use makedirs to create folders

# Preprocessing.py
## Exp-1
### Input
* train.csv of obs, sg5, sg7, sg9, ew2, ew3, ew4, and or.
* test.csv.
### Purpose
* Add positional encoding to the input train/test data.
### Task
* [x] Add hour and day positional encoding to NH$_{3}$N data.
### Output
* train_dataset.csv of obs, sg5, sg7, sg9, ew2, ew3, ew4, and or.
* test_dataset.csv.
---
## Exp-2
---
## Exp-3
## Checklist
* [x] Create only the hour and day positional encoding
* [x] Export to the right directories
# models.py
## Exp-1
### Purpose
* Generate available models for training.
### Tasks
* [ ] Random forest
* [x] CNN
  * src has been changed into [: ,: ,0 ] (1 feature)
  * In CNN, the src is required to be premuted ([N ,: ,features] instead of [: ,N ,features])
* [x] RNN
* [x] GRU
* [x] DNN
* [x] LSTM
## Exp-2
* [ ] DNN-5
* [ ] LSTM-5
* [ ] Attn-LSTM-5
* [ ] Transformer
---
## Exp-3
## Checklist
* [x] Check the input data into the models follows the right format stated by pytorch website.
* [x] Check the input, output size of the model.
 
# Dataloader.py
### Purpose
* Input the train and test dataset one by one.
---
### Checklist
## TrainDataset
* [ ] check the len__
* [ ] the input and target should only input with positional encoding of hour and day
## TestDataset
* [ ] The len needs to be change according to the length of the testdataset.
# helpers.py
### Purpose
* Write loss file save location
* Other functions
### Task
#### Directory related
* [ ] Modify the file saving path of log_loss
* [ ] Modify the file saving path of log_test_loss
* [ ] Modify the clean directory function according to new file saving location
#### Stability_test func
* [ ] Consider the output file saving directory according to different database.
# main_load.py
### purpose
* To input all the required parameters for main.py to run.
* Organize all the script and output directories of results.
### Task
* [ ] The ways to input different databases.
* [ ] loss saving path
* [ ] pred saving path
* [ ] model saving path
* [ ] hyperparameters
# main.py
### Purpose
* Input arguments and run with python.


