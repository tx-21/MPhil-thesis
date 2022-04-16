# RESULTS
## Exp-1 (NH$_{3}$N baseline model)
### Keys
* [ ] The benefit of data pre-processing by comparing validation and test loss.
* [ ] The selection of best model by comparing validation and test loss.
* [ ] Test data could be in poor quality.
* [ ] Show another test data results and compare the test and valid loss.

### Fig and table
* Exp 1
  
  |Pre-processing methods|Train date|Valid date|Test date|Algorithms|
  |:---|:---:|:---:|:---:|:---:|
  |obs|                   |               |              |CNN       |
  |sg5|                   |               |              |DNN       |
  |sg7|                   |               |              |RNN       |
  |sg9|12/13/2021—1/9/2022|1/10—l1/15/2022|1/16—1/22/2022|GRU       |
  |ew2|                   |               |              |LSTM      |
  |ew3|                   |               |              |          |
  |ew4|                   |               |              |          |
  |or |                   |               |              |          |
  Table: Traning parameters in Exp-1. {#tbl:id}

* result 1
  After sorting the test loss from the lowest to the highest, we observed that the test loss from lowest doesn't match with the valid loss from lowest.

  |Model-dataset|Test Loss (1/16—1/22)      |Valid loss (1/10—1/15)|
  |:---         |:---:                      |:---: |
  |GRU-sg7	    |**0.0383**	                |1.2508|
  |GRU-sg5	    |0.0385	                    |1.2644|
  |LSTM-ew3	    |0.0388	                    |**1.0796**|
  |LSTM-sg7	    |0.0388	                    |1.1804|
  |LSTM-sg5	    |0.0388	                    |1.2346|
  Table: Test and valid loss of NH$_{3}$N in Exp-1. {#tbl:id}


|Model-dataset|Validation Loss|Model-dataset    |Test loss|
|:---         |:---:          |:---             |:---:    |
|LSTM-ew3	    |1.0796         |__GRU-sg7__	    |0.0383   |
|LSTM-ew2	    |1.0969         |GRU-sg5	        |0.0385   |
|LSTM-ew4	    |1.1219         |__LSTM-ew3__	    |0.0388   |
|LSTM-sg7	    |1.1804         |__LSTM-sg7__	    |0.0388   |
|GRU-ew2	    |1.1891         |__LSTM-sg5	__    |0.0388   |
|GRU-ew3	    |1.2199         |__GRU-ew2__	    |0.0389   |
|LSTM-sg5	    |1.2346         |__GRU-ew4__	    |0.0391   |
|LSTM-obs	    |1.2366         |__LSTM-ew2__     |0.0392   |
|GRU-ew4	    |1.239          |__GRU-ew3__	    |0.0392   |
|GRU-sg7	    |1.2508         |__LSTM-ew4__	    |0.0395   |


Table: Valid and test loss from 1/16 to 1/22. {#tbl:2}

|Model-dataset|Validation Loss|Model-dataset    |Test loss|
|:---         |:---:          |:---             |:---:    |
|LSTM-ew3	    |1.0796         |__LSTM-ew3__	    |0.0158|
|LSTM-ew2	    |1.0969         |__LSTM-ew2__	    |0.0161|
|LSTM-ew4	    |1.1219         |__LSTM-ew4__	    |0.0163|
|LSTM-sg7	    |1.1804         |__LSTM-sg5__	    |0.0166|
|GRU-ew2	    |1.1891         |__GRU-ew3__	    |0.0167|
|GRU-ew3	    |1.2199         |__GRU-ew4__	    |0.0169|
|LSTM-sg5	    |1.2346         |__GRU-ew2__	    |0.0170|
|LSTM-obs	    |1.2366         |GRU-sg9	        |0.0174|
|GRU-ew4	    |1.239          |__LSTM-obs__	    |0.0175|
|GRU-sg7	    |1.2508         |LSTM-or	        |0.0177|


Table: Valid and test loss from 1/16 to 1/22. {#tbl:3}

| Reagent                  | Amount   |
| ------------------------ | -------- |
| Appropriate Buffer (10x) | 1x       |
| DNA                      | 50-500ng |
| Restriction Enzyme       | 1*U*     |
| Water                    | -        |

Table: {#tbl:restriction-generic} Schematic for restriction digestion with a single restriction enzyme. Some really long text that shows how the caption is formatted when it takes multiple lines.

## Exp-2

## Exp-5

## Exp-6
# Result
## sdfas
|Model-dataset|Validation Loss|
|:---         |:---: |
|LSTM-ew3	    |1.0796|
|LSTM-ew2	    |1.0969|
|LSTM-ew4	    |1.1219|

Table: Validation and test loss comparison from 1/16 to 1/22. {#tbl:2}

## asdf
|Model-dataset|Validation Loss|Model-dataset|Test loss|
|:---         |:---: |:---            |:---: |
|LSTM-ew3	    |1.0796|GRU-sg7	        |0.0383|
|LSTM-ew2	    |1.0969|GRU-sg5	        |0.0385|
|LSTM-ew4	    |1.1219|__LSTM-ew3__	  |0.0388|

Table: Validation and test loss comparison from 1/16 to 1/22. {#tbl:2}

![test](D:\MPhil-thesis-github-library\MPhil-thesis\Thesis\Chap 3_Materials and Methods\LSTM_1_pred_Step1.png){#fig:id width=50%}

Thanks, it works. But I have another problem now. My images are a little large, and when put in the same row they cannot fit into one slide. Is it possible to control the size of the image? Thanks, it works. But I have another problem now. My images are a little large, and when put in the same row they cannot fit into one slide. Is it possible to control the size of the image? Thanks, it works. But I have another problem now. My images are a little large, and when put in the same row they cannot fit into one slide. Is it possible to control the size of the image? Thanks, it works. But I have another problem now. My images are a little large, and when put in the same row they cannot fit into one slide. Is it possible to control the size of the image? 

![tesst](D:\MPhil-thesis-github-library\MPhil-thesis\Thesis\Chap 3_Materials and Methods\LSTM_1_pred_Step1.png){#fig:id width=50%}

Thanks, it works. But I have another problem now. My images are a little large, and when put in the same row they cannot fit into one slide. Is it possible to control the size of the image? Thanks, it works. But I have another problem now. My images are a little large, and when put in the same row they cannot fit into one slide. Is it possible to control the size of the image? Thanks, it works. But I have another problem now. My images are a little large, and when put in the same row they cannot fit into one slide. Is it possible to control the size of the image?
