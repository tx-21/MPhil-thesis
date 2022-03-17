## Slide contents
### Introduction
1. Remind the audience the goal of this project. (1 slide)
2. Mention the focus of creating a good machine learning models depends on several things.
    * Graph
3. Use bullet point from the alst slide and talk about the previous work. (transition: the previous works only cover .... except feature engineering, to build a better model, thus... )
4. Explain what is feature engineering 
5. Talk about my discovery, repetitive patterns (peak at the same time, resemble to residence time distributino from a plug flow), to propose that if all these can be turned into features, perhaps we can use a way to record the absolute timestamp of each measurement. The purpose of this is to tell the machine usually around 10~11 a.m., the ammonia concentration is the highest. 
6. objective
    * Using positional encoding as new feature engineering method.
    * Using a state-of-the-art deep learning model with positional encoding to forecast NH3.

### Results
#### Objective 1
1. Explain how I add positional encoding into our training data.
    * pictures and more visulizations
2. Remind the audience of how does DNN and LSTM work.
2. Compare DNN, DNN+pos, LSTM, LSTM+pos.
    > LSTM+pos demonstrate the best performance.
#### Objective 2
1. Explain why using the Transformer model
    * Compared to recursive neural networks, Transformer model can use attention mechanism to replace the passing hidden state function to achieve superior performance compared to RNNs.
2. Compare results of all the models

(Need to make up experiment for LSTM + ATTN?)
### plotting
1. Error bar plot of 
    * [x] train loss
    * [x] test loss
    * [x] rmse
    * [x] r2
2. positional encoding
    * [x] day
    * [x] hour

---
## Objective
1. Explain what I have accomplished in the previous presentation to remind them of my previous works.
    * I used the filters, and outlier removals, but these can't really explain which model peforms better.  
        * Show the previous results to prove this.
    * Explain in the previous works, how I predict step 2 forecast is by training the model to output 2 values, in the new project, the method has changed into training the model on predicting on the step 1 forecast only, and use the forecasted values as input of the model, then generate the step 2 forecast (model only does one thing, which is to forecast the value in the next forecast horizon).  
        * Limitaion, the new training method is a new model architecture, therefore, I haven't have time to optimize the model to compare with the former results.  
            >I should name the iterated versions of the models, so that I can compare the difference between the models in a clear way
2. Introduce the audience the new concepts, which are the positional encodings, and the attention mechanism.
    * Should mention how I design my experiment to prove these works on my data.
        * [ ] The attention mechanism with LSTM should be included, but only in LSTM. (need to look up for open source code)
3. Use the results to prove my hypothesis.

## Keys points
1. The model performance can be explained by engineered features with meaning.
    * The benefits of using position encoding can enhence the model performance.
        * Examples are MLP, LSTM
    * The benefits of using attention mechanism could outperform recursive neural networks.
    
### Visualization
* Using attention score layer graph (expect six attention layers can be shown) (explanable)
* The working mechanism of attention layers


## Future works
