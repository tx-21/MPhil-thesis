Study 1: Algorithm Selections and Data Pre-processing Methodologies in Developing Machine Learning Models for NH3N Forecasting.  
Study 2: To incoporate the use of domain knowledge in water/wastewater treatment and established methodology to construct a optimized model for NH$_{3}$N forecasting.

# Introduction
## Background
## Objectives 
### To evaluate baseline model performance in forecasting NH$_{3}$N by developing models with traditional and deep learning algorithms.
### To develop raw data cleaning methodologies for improved machine learning model performance of forecasting NH$_{3}$N.
### To explore feature engineering with the use of domain knowledge in wastewater treatment to creating new variables for optimizing model performance of forecasting NH$_{3}$N.
### To construct an optimized procedure of training a machine learning model for NH$_{3}$N forecasting. 
## Organization of the thesis
# Literature Review
## Water quality forecasting in wastewater treatment plant
### Tools and technologies for parameter forecasting in wastewater treatment plant
## Machine learning models for water quality forecasting
### Introduction to time-series data
### Machine learning models and comparison 
### Review of existing cases of applying machine learning for water quality forecasting
## Techniques for improving model forecasting performance
### Data pre-processing with smoothing and outlier removal 
### Implementation of weight regularization to avoid model overfittings
### Other regularization methods to avoid model overfittings
# Materials and methods
## Wastewater treatment plant description
### Treatment processes
### Historical water quality data
### Reclaimed water standard 
## Data collection and preparation
### NH3-N data monitoring and collection
### Data cleaning and pre-processing
#### Data smoothing with Savitzky-Golay filter

#### Data smoothing with EMCA filter
#### Outlier detection and removal
### Data transformation
#### Split of Train/valid/test dataset 
## Architecture design of the selected baseline models
### Model A (LSTM)
### Model B (DNN)
### ……
## Implementation of regularization on machine learning models
### Early-stopping 
### Dropout
### Weight regularization
## Proposed time series forecasting workflow
# Results and Discussion
## Comparisons of forecast accuracy in statistical and machine learning models
## The effect of data cleaning on forecast accuracy 
### Data smoothing
### Outlier removal
## The effect of regularization techniques on forecast accuracy
### Early-stopping
### Dropout
### Weight regularization
## The effect of input training datasets on the stability of forecast models
### Selection of the data training size
### Update input training dataset with up-to-date data
### Cross-validation

# Conclusions and recommendations 
