import numpy as np

from adaboost.adaboost_mrt import AdaboostMRT
from adaboost.error_functions import *
from typing import Tuple
from sklearn.neural_network import MLPRegressor


def generate_sample_line_data(min_x: float, max_x: float, n_items:int, noise_var:float)->Tuple[np.array, np.array]:
    x = np.random.uniform(min_x, max_x, n_items)
    y = x*5 + np.random.normal(0, noise_var, n_items)
    return x,y


if __name__=='__main__':
    x_train, y_train = generate_sample_line_data(-5,5,200, 0.2)
    x_test, y_test = generate_sample_line_data(-5,5,100, 0.2)

    # train Adaboost.MRT
    amrt = AdaboostMRT(base_learner=MLPRegressor, iterations=2)
    amrt.fit(x_train,y_train,N=100,phi=0.1,n=1, hidden_layer_sizes = (20,20), max_iter=700)

    # Apply to sample data
    y_test_predict = amrt.predict(x_test)
    print(np.mean(np.abs(y_test_predict - y_test)))
    print('Complete')

