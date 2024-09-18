<<<<<<< HEAD
# Dynamic Weighting of Source and Target Knowledge to Enhance Transfer Learning for Hyperparameter Optimization

This code reproduces the method of the paper 
[Dynamic Weighting of Source and Target Knowledge to Enhance Transfer Learning for Hyperparameter Optimization]

Four datasets of precomputed evaluations are available (DeepAR, FCNET, XGBoost and nas_bench102). The first three 
 are taken from this [repo](https://github.com/icdishb/hyperparameter-transfer-learning-evaluations), the last 
 one was generated from [NAS-bench-102](https://github.com/Debrove/NAS-Projects)).

This is an reimplementation from scratch of the method as the initial implemented depended on proprietary components. 
As such, the results differ and may be slightly worse given that different frameworks were used (botorch,
 pytorch for the GP in particular) but we made sure the reimplementation is reasonably close, see below for a 
 comparison with the reimplementation and our initial implementation.
 


## How to run

### setup interpreter

I recommend using the conda environment used for the evaluations is given in `environment-GC3P.yml`.
```
conda env create -f environment-GC3P.yml
conda activate GC3P
```

The requirements.txt is done manually and may give you something similar with less guarantees.

### figure and table generation

Regenerate figures and tables.
```
python experiments/ADTM.py
python experiments/plot.py
```
ADTM.py will get all the baseline and my method results on a dataset once, and plot.py will only draw a line graph of one result file at a time.


### benchmark 

All precomputed evaluations are placed in 
src/blackbox/offline_evaluations
DeepAR.csv.zip
FCNET.csv.zip
XGBoost.csv.zip
nas_bench102.csv.zip

To benchmark some methods on a given dataset quickly, see `benchmark_example.py`. This will run the optimizers
and plot their convergence distribution.



=======

