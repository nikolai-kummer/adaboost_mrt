import math
import numpy as np
import warnings

from numpy.core.fromnumeric import shape

from adaboost.adaboost_mrt import AdaboostMRT
from adaboost.error_functions import *
from typing import Tuple
from sklearn.exceptions import ConvergenceWarning,DataConversionWarning
from sklearn.neural_network import MLPRegressor


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


def generate_friedman1_data(n_items:int, noise_var:float=1.0)->Tuple[np.array, np.array]:
    x=np.random.uniform(0,1, size=(n_items, 10))
    y = 10*np.sin(math.pi*x[:,0]*x[:,1])+20*(x[:,2]-0.5)**2 + 10*x[:,3] + 5*x[:,4]
    return x,y + np.random.normal(0, noise_var, size=y.shape)



if __name__=='__main__':
    noise_amplitude = 2
    x_train, y_train = generate_friedman1_data(5000, noise_amplitude)
    x_test, y_test = generate_friedman1_data(1000, 0.0)

    # train Adaboost.MRT
    n_iterations = 5
    amrt = AdaboostMRT(base_learner=MLPRegressor, iterations=n_iterations)
    amrt.fit(x_train,y_train,N=2000,phi=0.3,n=2, hidden_layer_sizes = (10), max_iter=900, verbose=True)

    # Apply to sample data
    for idx in range(0, n_iterations):
        y_test_predict = amrt.predict_individual(x_test, list(range(0,idx+1)))
        print(f"First {idx+1} learners, RMSE: {np.sqrt(((y_test_predict - y_test) ** 2).mean())}")

    y_test_predict = amrt.predict(x_test)
    print(f"Full Ensemble learners, RMSE: {np.sqrt(((y_test_predict - y_test) ** 2).mean())}")
    print('Complete')

