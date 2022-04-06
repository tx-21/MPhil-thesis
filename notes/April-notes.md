# Goal
* Write code for exp 1, which is to select the baseline model.
* Write code for exp 2, which is to compare the model improvement after applying feature engineering.
* Write code for exp 3, which is to apply the established methodology to run with colour data.
  * Run with colour to predict colour.
  * Run with NH$_{3}$N + colour to predict NH$_{3}$N
  * Run with NH$_{3}$N + colour to predict colour.
# Deadline in April
* 4/22 Group presentation
# Progress
## 4/5
### raw-data-pre-processing.ipynb
* Turn obs into three SG filters, 3 EWMA filters, and OR.
  * The selection of three different window is see how does the model performance react to the degree of the window.
* Only data for Exp-1 is exported, for Exp-2 ~ 3 need to be decided later.
### models.py
* CNN, RNN, GRU, DNN, LSTM have been written.
  * src has been changed into [:,:,0] (1 feature)
  * In CNN, the src is required to be premuted ([N,:,features] instead of [:,N,features])
* Attn LSTM has not been revised.
* Random forest cannot run with deep learning models (since it doesn't need to update it's weight), therefore, an individual script is required to run on RF.
### Preprocessing.py
* All eight datasets are transformed into train_dataset.csv in Exp-1.


