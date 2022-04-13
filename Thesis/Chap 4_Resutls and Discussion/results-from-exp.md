# Exp-1
## GRU
* All the data-proprocessing methods improved the model performance.
* Obs has the lowest std.
* Obs has the highest rmse.
* sg7 has lowest rmse.
## RNN
* Obs has the highest rmse.
* RNN-sg5 has std of 0.0002 compared to others (0.0028~0.01)
* sg7 has lowest rmse.
## LSTM
* Obs has the highest rmse.
* or has lowest rmse and std.
## DNN
* All rmse are close to each other, ew3 has the lowest.
## CNN
* sg5 and sg7 share similar performance (also he lowest rmse)
* beside sg7 and sg7, the algorithm is extremely unstable.
## RF
* Most RF models have lower rmse compared to DNN models, and all the RF std are lower than DNN.
## Conclusion
* For recursive neural networks, all the pre-processing methods can improve the baseline model performance.
* The performance of RF and DNN don't improve with data pre-processing methods.
* The best model in sequence are:
  * GRU sg7, 0.2090
  * RNN sg7, 0.2173
  * LSTM or, 0.2185
