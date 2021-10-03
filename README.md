# Adaboost.MRT: Boosting regression for multivariate estimation

This repo is a python library with code for Adaboost.MRT. An old matlab version was also added for archive purposes. 

Adaboost.MRT is an ensemble multivariate regression extension of Adaboost.RT. It boosts the accuracy of a number of weak learners to create a single strong ensemble predictor.


***Note:*** The original work outlined in the paper was done in Matlab and this git repo is an implementation of that procedure in python. Results may not be the same as the implementation of the base learners is different in Matlab and Python.

## Sample Usage
The following is some sample code on how to run predictions:

```python
import numpy as np
from adaboost.adaboost_mrt import AdaboostMRT
from sklearn.neural_network import MLPRegressor

# Generate sample data
x = np.random.uniform(size=(500,5))
y = np.random.uniform(size=(500,2))

# Create Adaboost.MRT, fit to sample data, and run a prediction 
amrt = AdaboostMRT(base_learner=MLPRegressor, iterations=10)
amrt.fit(x,y,N=100,phi=[0.1,0.2],n=2, hidden_layer_sizes = (20,10), verbose=True)
prediction = amrt.predict(x)
```


## Citations

Reconstructed based on the work presented in the following paper:

Kummer, Nikolai, and Homayoun Najjaran. "Adaboost. MRT: Boosting regression for multivariate estimation." Artif. Intell. Res. 3.4 (2014): 64-76.

