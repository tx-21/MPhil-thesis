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
