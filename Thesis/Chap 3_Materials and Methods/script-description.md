# Ultimate goal
* 4/14
  * since test and valid data (12/23-1/15) look much different to the test (1/16-1/22), 




# Terminal
* Exp 2
  * Running with lr=1e-04 (to see if we can have gru_3 with best performance), i will be generating new test data (10/10-10/16) test again verify the model performance. 
# Exp function description
## Exp 1
* 
# Exp steps
## Exp 1
* Use scheduler with fixed epoch and lr to find the optimal combination of dataset and algorithms.
* result-1, factor=0.5, patience=10, epoch=100, lr=5e-05, Exp_num=3, GRU-sg7 = 0.207
## Exp 2
# ongoing
* using the saved best model to predict the new test data (10/10-10/16)
# done
* result-1, factor=0.5, patience=10, epoch=100, lr=5e-05, Exp_num=3 
  * GRU sg7 (0.207) > GRU attn (0.2185) > GRU-7 sg7 (0.2334)
    * attn perform poorer than GRU sg7.... attn is not helping.... 
    * I compared the prediction plots of gru, gru-anchor, and gru attn, at peak time, all showed not improved any.
* result-2, factor=0.5, patience=10, epoch=100, lr=1e-04, Exp_num=3
  * In result-2, all the anchor models are included, and all the input are changed from 5 to 3 (nh3 + hours only)
* result-3, factor=0.5, patience=10, epoch=100, lr=5e-05, Exp_num=3
* result-4, factor=0.1, patience=10, epoch=100, lr=1e-04, Exp_num=3
* result1 and 3 is the same, but the GRU-1 model screwed up in result1

# Final version
* Hyperparameters
  * lr=1e-04
  * epoch=100
  * factor=0.1
  * patience=10
  * Exp_num=5
* Exp 1
  * RNN
  * GRU
  * LSTM
  * CNN
  * DNN
  * RF (Exp 1.1)
* Exp 2
  * Select three models with the specific pro-processing method.
  * GRU
  * RNN
  * LSTM
  * GRU attn
  * RNN attn
  * LSTM attn
  * GRU anchor
  * RNN anchor
  * LSTM anchor