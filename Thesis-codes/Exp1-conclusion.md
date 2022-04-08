# Discussion
## Determine the top three models for futher optimization
* Should only compare the observed dataset, the reasons are the following
    * The optimization of the model performance is left in the futher exp. 
    * The pre-processed methods are here only to assist in determine the best baseline model performance for the purpose of reference.
## Determine the best pre-processing method
* The metric for evaluating the pre-processing method should be:
    * the increase/decrease of std
    * the increase/decrease of mean
* We can include the stability of the model performance provided by data preprocessing.

# Conclusion
## Obs (select top 3 model)
* Mean
    * DNN>RNN>LSTM
* std
    * LSTM>RNN>DNN
## DNN
* Mean
    * sg5>sg9>ew2
* std
    * sg9>sg5>ew2
## RNN
* Mean
    * sg7>sg5>ew4
* std
    * ew4>sg7>sg5
## LSTM
* Mean
    * sg5>sg9>sg7
* std
    * sg5>sg9>sg7
## Baseline model performance
* sg5 DNN rmse = 0.4211 0.063