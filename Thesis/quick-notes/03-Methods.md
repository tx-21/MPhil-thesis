# Model
## *Wu et al. (2020)*
### Problem Description
We formuate ILI forecasting as a supervised ma- chine learning task. Given a time series contain- ing $N$ weekly data points $x_{t−N+1},...,x_{t−1},x_t$, for M- step ahead prediction, the input $X $of the supervised ML model is $x_{t−N} +1 , ..., x_{t−M}$ , and the output $Y$ is $x_{t−M+1},x_{t−M+2},...,x_t$. Each data point $x_t$ can be a scalar or a vector containing multiple features.
### Data
To produce a labeled dataset, we used a fixed-length sliding time window approach (Figure 5) to construct $X, Y$ pairs for model training and evaluation. Before applying the slid- ing window to get features and labels, we perform min-max scaling on all the data with the maximum and minimum val- ues of training dataset. We then run a sliding window on the scaled training set to get training samples with features and labels, which are the previous N and next M observations respectively. Test samples are also constructed in the same manner for model evaluation. The train and test split ratio is 2:1. Training data from different states are concatenated to form the training set for the global model.
### Transformer Model
#### 1 MODEL ARCHITECTURE
1. Encoder  
The encoder is composed of an input layer, a po- sitional encoding layer, and a stack of four identical encoder layers. The input layer maps the input time series data to a vector of dimension $d_{model}$ through a fully-connected network. This step is essential for the model to employ a multi- head attention mechanism. Positional encoding with sine and cosine functions is used to encode sequential information in the time series data by element-wise addition of the input vector with a positional encoding vector. The resulting vector is fed into four encoder layers. Each encoder layer consists of two sub-layers: a self-attention sub-layer and a fully-connected feed-forward sub-layer. Each sub-layer is followed by a normalization layer. The encoder produces a $d_{model}$-dimensional vector to feed to the decoder.  

2. Decoder  
We employ a decoder design that is similar to the original Transformer architecture (Vaswani et al., 2017). The decoder is also composed of the input layer, four identi- cal decoder layers, and an output layer. The decoder input begins with the last data point of the encoder input. The input layer maps the decoder input to a $d_{model}dimensional vector. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer to apply self- attention mechanisms over the encoder output. Finally, there is an output layer that maps the output of last decoder layer to the target time sequence. We employ look-ahead masking and one-position offset between the decoder input and tar- get output in the decoder to ensure that prediction of a time series data point will only depend on previous data points
#### 2 TRAINING
1. Training Data and Batching  
In a typical training setup, we train the model to predict 4 future weekly ILI ratios from 10 trailing weekly datapoints. That is, given the encoder input (x1, x2, ..., x10) and the decoder input (x10, ..., x13), the decoder aims to output (x11, ..., x14). A look-ahead mask is applied to ensure that attention will only be applied to datapoints prior to target data by the model. That is, when predicting target (x11, x12), the mask ensures attention weights are only on (x10, x11) so the decoder doesn’t leak information about x12 and x13 from the decoder input. A minibatch of size 64 is used for training.
2. Optimizer
We used the Adam optimizer (Kingma & Ba, 2015) with β1 = 0.9, β2 = 0.98 and ε = $10^{−9}$. A custom learning rate with following schedule is used:  
$lrate =d0.5 ∗ min(step num0.5, model step num ∗ warmup steps−1.5)$
3. Regularization
We apply dropout techniques for each of the three types of sub-layers in the encoder and decoder: the self-attention sub-layer, the feed-forward sub-layer, and the normalization sub-layer. A dropout rate of 0.2 is used for each sub-layer.
#### 3 EVALUATION
In evaluation, labeled test data are constructed using a fix- length sliding window as well. One-step ahead prediction is performed by the trained Transformer model. We computed Pearson correlation coefficient and root-mean-square errors (RMSE) between the actual data yi and the predicted value yˆ .
### ARIMA, LSTM, and seq2deq models 
(mine would be MLP/LSTM)
#### ARIMA
#### LSTM
#### MLP
