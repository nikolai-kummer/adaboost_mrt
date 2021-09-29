import numpy as np
import warnings

from adaboost.adaboost_mrt import AdaboostMRT
from adaboost.error_functions import *
from typing import Tuple
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning) # ignore since we are artifically constraining convergence

def generate_sample_line_data(min_x: float, max_x: float, n_items:int, noise_var:float)->Tuple[np.array, np.array]:
    x = np.random.uniform(min_x, max_x, n_items)
    y = x*5 + np.random.normal(0, noise_var, n_items)
    return x,y


if __name__=='__main__':
    x_train, y_train = generate_sample_line_data(-5,5,200, 2)
    x_test, y_test = generate_sample_line_data(-5,5,100, 0.0)

    # train Adaboost.MRT
    n_iterations = 10
    amrt = AdaboostMRT(base_learner=MLPRegressor, iterations=n_iterations)
    amrt.fit(x_train,y_train,N=100,phi=0.01,n=2, hidden_layer_sizes = (20,20), max_iter=400, verbose=True)

    # Apply to sample data
    for idx in range(0, n_iterations):
        y_test_predict = amrt.predict_individual(x_test, list(range(0,idx+1)))
        print(f"First {idx+1} learners, RMSE: {np.sqrt(((y_test_predict - y_test) ** 2).mean())}")

    y_test_predict = amrt.predict(x_test)
    print(f"Full Ensemble learners, RMSE: {np.sqrt(((y_test_predict - y_test) ** 2).mean())}")
    print('Complete')

