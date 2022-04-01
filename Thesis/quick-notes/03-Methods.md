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
## Patel (2021)
### What is Feature Engineering
* Benchmark : A Benchmark Model is the most user-friendly, dependable, transparent, and interpretable model against which you can measure your own. It’s a good idea to run test datasets to see if your new machine learning model outperforms a recognised benchmark. These benchmarks are often used as measures for comparing the performance between different machine learning models like neural networks and support vector machines, linear and non-linear classifiers, or different approaches like bagging and boosting. To learn more about feature engineering steps and process, check the links provided at the end of this article. Now, let’s have a look at why we need feature engineering in machine learning.
* Feature Creation (adding colour data): Creating features involves creating new variables which will be most helpful for our model. This can be adding or removing some features. As we saw above, the cost per sq. ft column was a feature creation.
* Transformations (transform datetime data to positional encoding): Feature transformation is simply a function that transforms features from one representation to another. The goal here is to plot and visualise data, if something is not adding up with the new features we can reduce the number of features used, speed up training, or increase the accuracy of a certain model.
### Feature Engineering Techniques for Machine Learning
#### 1.Imputation
When it comes to preparing your data for machine learning, missing values are one of the most typical issues. Human errors, data flow interruptions, privacy concerns, and other factors could all contribute to missing values. Missing values have an impact on the performance of machine learning models for whatever cause. The main goal of imputation is to handle these missing values. There are two types of imputation :
#### 2.Handling Outliers
Outlier handling is a technique for removing outliers from a dataset. This method can be used on a variety of scales to produce a more accurate data representation. This has an impact on the model’s performance. Depending on the model, the effect could be large or minimal; for example, linear regression is particularly susceptible to outliers. This procedure should be completed prior to model training. The various methods of handling outliers include:
* Removal: Outlier-containing entries are deleted from the distribution. However, if there are outliers across numerous variables, this strategy may result in a big chunk of the datasheet being missed.
*  Replacing values: Alternatively, the outliers could be handled as missing values and replaced with suitable imputation.
#### 5.Scaling
Feature scaling is one of the most pervasive and difficult problems in machine learning, yet it’s one of the most important things to get right. In order to train a predictive model, we need data with a known set of features that needs to be scaled up or down as appropriate.
* Normalization : All values are scaled in a specified range between 0 and 1 via normalisation (or min-max normalisation). This modification has no influence on the feature’s distribution, however it does exacerbate the effects of outliers due to lower standard deviations. As a result, it is advised that outliers be dealt with prior to normalisation.
* Standardization: Standardization (also known as z-score normalisation) is the process of scaling values while accounting for standard deviation. If the standard deviation of features differs, the range of those features will likewise differ. The effect of outliers in the characteristics is reduced as a result. To arrive at a distribution with a 0 mean and 1 variance, all the data points are subtracted by their mean and the result divided by the distribution’s variance.

# Baseline model selection
## The purpose of creating baseline models
There are few requirements for a good baseline model:
* Baseline model should be simple. Simple models are less likely to overfit. If you see that your baseline is already overfitting, it makes no sense to go for more complex modelling, as the complexity will kill the performance.
* Baseline model should be interpretable. Explainability will help you to get a better understanding of your data and will show you a direction for the feature engineering.
* 