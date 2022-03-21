# Materials and methods
## Wastewater treatment plant description
### Treatment processes
### Historical water quality data
### Reclaimed water standard
### Problem Description
Describe how to solve the problem by using supervised machine learning task...
## Data collection and preparation
### NH3-N data monitoring and collection
### Data preparation process
#### Data smoothing
#### Outlier detection and removal
#### Feature engineering
@patelWhatFeatureEngineering2021
What is Feature Engineering
Feature engineering is a machine learning technique that leverages data to create new variables that aren’t in the training set. It can produce new features for both supervised and unsupervised learning, with the goal of simplifying and speeding up data transformations while also enhancing model accuracy. Feature engineering is required when working with machine learning models. Regardless of the data or architecture, a terrible feature will have a direct impact on your model.



#### Split of Train/valid/test dataset 
## Machine learning models
### MLP
#### Model Architecture
### LSTM
#### Model Architecture
### Transformer
#### Model Architecture
##### Decoder
@wuDeepTransformerModels2020
We employ a decoder design that is similar to the original Transformer architecture (Vaswani et al., 2017). The decoder is also composed of the input layer, four identi- cal decoder layers, and an output layer. The decoder input begins with the last data point of the encoder input. The input layer maps the decoder input to a $d_{model}dimensional vector. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer to apply self- attention mechanisms over the encoder output. Finally, there is an output layer that maps the output of last decoder layer to the target time sequence. We employ look-ahead masking and one-position offset between the decoder input and tar- get output in the decoder to ensure that prediction of a time series data point will only depend on previous data points
##### Attention mechanism
## Training
### Data transformation
To produce a labeled dataset, we used a fixed-length sliding time window approach (Figure 5) to construct $X, Y$ pairs for model training and evaluation. Before applying the slid- ing window to get features and labels, we perform min-max scaling on all the data with the maximum and minimum val- ues of training dataset. We then run a sliding window on the scaled training set to get training samples with features and labels, which are the previous N and next M observations respectively. Test samples are also constructed in the same manner for model evaluation. The train and test split ratio is 2:1. Training data from different states are concatenated to form the training set for the global model.  
  
In a typical training setup, we train the model to predict 4 future weekly ILI ratios from 10 trailing weekly datapoints. That is, given the encoder input (x1, x2, ..., x10) and the decoder input (x10, ..., x13), the decoder aims to output (x11, ..., x14). A look-ahead mask is applied to ensure that attention will only be applied to datapoints prior to target data by the model. That is, when predicting target (x11, x12), the mask ensures attention weights are only on (x10, x11) so the decoder doesn’t leak information about x12 and x13 from the decoder input. A minibatch of size 64 is used for training.
### Implementation of regularization on machine learning models
#### Early-stopping 
#### Dropout
#### Weight regularization
## Evaluation
In evaluation, labeled test data are constructed using a fix- length sliding window as well. One-step ahead prediction is performed by the trained Transformer model. We computed Pearson correlation coefficient and root-mean-square errors (RMSE) between the actual data yi and the predicted value yˆ .