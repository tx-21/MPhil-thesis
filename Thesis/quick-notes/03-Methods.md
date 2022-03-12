## Model
---
@\cite{wuDeepTransformerModels2020}]  
### Problem Description  
We formuate ILI forecasting as a supervised ma- chine learning task. Given a time series contain- ing $N$ weekly data points $x_{t−N+1},...,x_{t−1},x_t$, for M- step ahead prediction, the input $X $of the supervised ML model is $x_{t−N} +1 , ..., x_{t−M}$ , and the output $Y$ is $x_{t−M+1},x_{t−M+2},...,x_t$. Each data point $x_t$ can be a scalar or a vector containing multiple features.
### Data  
To produce a labeled dataset, we used a fixed-length sliding time window approach (Figure 5) to construct $X, Y$ pairs for model training and evaluation. Before applying the slid- ing window to get features and labels, we perform min-max scaling on all the data with the maximum and minimum val- ues of training dataset. We then run a sliding window on the scaled training set to get training samples with features and labels, which are the previous N and next M observations respectively. Test samples are also constructed in the same manner for model evaluation. The train and test split ratio is 2:1. Training data from different states are concatenated to form the training set for the global model.
### Transformer Model
#### Model Architecture
Consisting of decoder layers...  
__Encoder__ ..(explain)  
The encoder is composed of an input layer, a po- sitional encoding layer, and a stack of four identical encoder layers. The input layer maps the input time series data to a vector of dimension $d_{model}$ through a fully-connected network. This step is essential for the model to employ a multi- head attention mechanism. Positional encoding with sine and cosine functions is used to encode sequential information in the time series data by element-wise addition of the input vector with a positional encoding vector. The resulting vector is fed into four encoder layers. Each encoder layer consists of two sub-layers: a self-attention sub-layer and a fully-connected feed-forward sub-layer. Each sub-layer is followed by a normalization layer. The encoder produces a $d_{model}$-dimensional vector to feed to the decoder.
__Decoder__ .. (explain)  
We employ a decoder design that is similar to the original Transformer architecture (Vaswani et al., 2017). The decoder is also composed of the input layer, four identi- cal decoder layers, and an output layer. The decoder input begins with the last data point of the encoder input. The input layer maps the decoder input to a $d_{model}dimensional vector. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer to apply self- attention mechanisms over the encoder output. Finally, there is an output layer that maps the output of last decoder layer to the target time sequence. We employ look-ahead masking and one-position offset between the decoder input and tar- get output in the decoder to ensure that prediction of a time series data point will only depend on previous data points

