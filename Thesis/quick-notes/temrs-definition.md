## Terms
* sequence-aligned models [@\cite{wuDeepTransformerModels2020}]

## Models
### Statistical models
1. ARIMA
    >Box-Jenkins ARIMA (Auto-Regressive Inte- grated Moving Average) is another popular approach to modeling dynamical systems. ARIMA models the observed variable $x_t$ and assumes $x_t$ can be decomposed into trend, seasonal and irregular components. Instead of modeling these components separately, Box and Jenkins had the idea of differencing the time series xt in order to eliminate trend and seasonality. The resulting series is treated as station- ary time series data and is modeled using combination of its lagged time series values (“AR”) and moving average of lagged forecast errors (“MA”). An ARIMA model is typically specified by a tuple $(p, d, q)$, where p and q de- fine the orders of AR and MA, and d specifies the order of differencing operation.
    >
2. Time Delay Embedding
### Machine learning models
#### Sequence Models
1. Recurrent Neural Networks
2. LSTM
    >While RNN has internal memory to process se- quence data, it suffers from gradient vanishing and explod- ing problems when processing long sequences. Long Short- Term Memory (LSTM) networks were specifically devel- oped to address this limitation (Hochreiter & Schmidhuber, 1997). LSTM employs three gates, including an input gate, forget gate and output gate, to modulate the information flow across the cells and prevent gradient vanishing and explosion.
    >

