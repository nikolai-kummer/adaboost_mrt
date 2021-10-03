import math
import numpy as np
import warnings

from adaboost.adaboost_mrt import AdaboostMRT
from adaboost.error_functions import *
from typing import Tuple
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.neural_network import MLPRegressor


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


def generate_swiss_roll_data(n_items:int, noise_var:float)->Tuple[np.array, np.array]:
    """Generates a swiss roll dataset

    Args:
        n_items (int): number of samples to generate
        noise_var (float): noise variance to add to data

    Returns:
        Tuple[np.array, np.array]: [description]
    """
    d = np.random.uniform(0, 10, n_items)
    theta = np.random.uniform(0, 4*math.pi, n_items)
    
    x_i = theta*np.cos(theta)
    y_i = d
    z_i = theta*np.sin(theta)

    x = np.stack((d,theta), axis=1)
    y = np.stack((x_i, y_i, z_i), axis=1)
    y = y + np.random.normal(0, noise_var, size=y.shape)
    return x,y


if __name__=='__main__':
    noise_amplitude = 1
    x_train, y_train = generate_swiss_roll_data(500, noise_amplitude)
    x_test, y_test = generate_swiss_roll_data(200, 0.0)

    # train Adaboost.MRT
    n_iterations = 10
    amrt = AdaboostMRT(base_learner=MLPRegressor, iterations=n_iterations)
    amrt.fit(x_train,y_train,N=200,phi=[0.6, 0.2, 0.6],n=2, hidden_layer_sizes = (3), max_iter=400, verbose=True)

    # Apply to sample data
    for idx in range(0, n_iterations):
        y_test_predict = amrt.predict_individual(x_test, list(range(0,idx+1)))
        print(f"First {idx+1} learners, RMSE: {np.sqrt(((y_test_predict - y_test) ** 2).mean())}")

    y_test_predict = amrt.predict(x_test)
    print(f"Full Ensemble learners, RMSE: {np.sqrt(((y_test_predict - y_test) ** 2).mean())}")
    print('Complete')

