# Discussion
## Observation
### 1-Rank the
* For NH$_{3}$N forecasting, from lowest loss in prediction of NH$_{3}$N in 1016-1022 is: only compare the test loss, not the RMSE.    
  * 0116-0122
    * Validation loss (Mean Squared Error)
      * LSTM-ew3, LSTM-ew2, LSTM-ew4, LSTM-sg7, GRU-ew2, GRU-ew3
    * FC1 RMSE (1/6) (0/3)
      * GRU-sg7, GRU-sg9, GRU-sg7, LSTM-or, RNN-sg5, GRU-ew4
    * Test loss (3/6) (0/3)
      * GRU-sg7, GRU-sg5, LSTM-sg7, LSTM-sg5, LSTM-ew3, GRU-ew2
  * 1010-1016
    * FC1 RMSE (4/6) (2/3)
      * LSTM-ew3, LSTM-ew4, GRU-sg9, LSTM-sg5, LSTM-sg7, LSTM-ew2
    * Test loss (4/6) (3/3)
      * LSTM-ew3, LSTM-ew2, LSTM-ew4, LSTM-sg5, GRU-ew3, GRU-ew4
  * 1022-1028
    * FC1 RMSE (1/6) (0/3)
      * GRU-obs, LSTM-obs, GRU-ew2, LSTM-ew2, RNN-obs, RNN-ew2
    * Test loss (2/6) (0/3)
      * LSTM-obs, GRU-obs, GRU-ew2, LSTM-ew2, RNN-obs, GRU-or

## Conclusion
* The comparison between validation and RMSE rank help us to know whether our test dataset is contaminated.