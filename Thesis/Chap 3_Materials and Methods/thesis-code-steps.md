# Exp 1
## Purpose
The purpose of Exp 1 is to use different preprocess methods to find the best performance model, which is also called the baseline model.
## Step description
1. Collect NH$_{3}$N and colour data
   * will be selected from 12/23 to 1/22.
   * The selection of this time period is due to this is the only available time period for colour, other period we have encountered some issues, which are the followings:
   * Power black out
   * Disconnect of the instrument
   * Abnormal event occured from the site
2. Perform data pre-processing
   * Remove the extreme values
   * Linear interpolation
   * data pre-processing
     * Obs/SG1/SG2/EWMA1/EWMA2/OR
3. Data transformation
   * Train/test — 80:20 (in creating baseline model, overfitting won't be a big problem, so let's do valid dataset in the optimization part)
    * Training step in
       * 24hr/12hr/6hr
     * Training step out
       * 3hr
4. Sort the prepared and format files together
5. Write the model algorithms
   * RF
   * DNN
   * CNN
   * RNN
   * LSTM
6. Write the output form of the exp results
   * RMSE/R2
     * MEAN/STD/UB/LB
   * Predicted results
   * Trained model
7. Write the codes for ploting
   * For making decision
     * Train with box plot
     * Test with box plot
   * For visulization
     * R2 in the best model plotting
     * Save all the model parameters
       * Extract step 1 only
       * Extract step 2 only
       * Extract step 3 only
8. Use teacher focing to train
9. Inference
# Exp 2
## Purpose
The purpose of __Exp 2__ is to use the top three performance models, using the optimized pre-processed method to train with feature engineering (positional encoding).
## Step description
1. Copy the optimized training dataset to new folder.
2. Append positional encoding to three datasets. (LSTM DNN RF)
3. Prepare the algorithm (see if i can run smoothly with the last two features are removed)
   * RF-5
   * DNN-5
   * LSTM-5
   * Transformer
4. Data transformation
   * Need to split into Train/valid/test as 70:15:15
5. Use tensorboard also other regularization method?
6. Plot the train/valid loss curve
7. Same as Exp 1 no. 6,7
8. Use teacher focing and scheduled sampling
9. Inference
10. Generate the final output and compare with baseline model.
# Exp 3
## purpose
The purpose of Exp 3 is to test whether the methodology I built for NH$_{3}$N can be applied to predicting colour, also using colour and NH$_{3}$N combined to predict either of it.
## Step description
1. Create a new training database of colour.
2. Create a new training database of NH$_{3}$N + colour