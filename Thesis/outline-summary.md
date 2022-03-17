# Listing the key targets in each chapter
## Chap 1 Introduction
## Chap 2 Literature Review
1. The forecasting water quality works have been done in the industry.
2. The recent works of water reclamation in wastewater treatment plant.
3. 

## Chap 3 Materials and Methods
1. Data collection 
    * Collecting __ammonia data__ from the on-line ammonia sensor
    * Collecting __colour data__ from the modified colour spectrophotometer
2. Data discovery and profiling
    * Identifying patterns, relationships, and other attributes in the data, including the inconsistencies, anomalies, missing values and other issues so they can be addressed.
3. Data cleansing
    * Smoothing data with polynomial curve and exponentially moving average.
    * Identifying abnormal days with peak detection analysis.
4. Data structuring
    * Organize data into the accessible form to python script.
5. Data transformation and enrichment
    * Adding ammonia/colour data as new features.
    * Creating new features (feature engineering).
    * Transforming data into training/testing format.
6. Model training
    * Select baseline models.
        * RF, DNN, LSTM, Transformer
    * Train model with different training input.
        * 2 filters.
        * Remove abnormal days
        * With/without colour/ammonia features.
        * With/without engineering features.
7. Model evaluation
    * Compare results in RMSE, R2, train/test loss.
    * Show the attention layer map.
8. Results
    * Generate the optimized method to forecast ammonia/colour.
## Chap 4 Results and Discussion
## Chap 5 Conclusions and Recommendations
