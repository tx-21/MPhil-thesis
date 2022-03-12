## Requires experiment
---
### Batch size
* [ ] The influence of batch size on time series forecasting
### Sheduled learning rate (need to consider the neccessity of using the shceduled lrn rate)
* $lrate = d^{0.5}_{model}*min(step\_num^{0.5}, step\_num*warmup\_steps^{-1.5})$
### Droppout rate
### Pearson correlation
Wu et al. (2020) suggests we can use pearson correlation to compare the performance between different models.
* 0.2 is used by Wu et al. (2020).
## Requires to search for information
---
### 1. Explanable models
* [ ] Using some statistical models to provide some good explanation.
### 2. Attetnion-based techniques
* [ ] multi-channel LSTM neural networks with attention layer  
(Attention- based recurrent neural network for influenza epidemic prediction.)
* [ ] seq-seq model with attention mechanism  
(Sequence to sequence with attention for influenza prevalence predic- tion usinggoogle trends.)
### 3. "Forget" problem of LSTM neural networks
* [ ] When explaining LSTM for it's drop of long-term dependencies, how do we prove this point? If conducting experiments with different input length will help to explain? 
