import numpy as np
import random
import warnings
import xgboost as xgb

from adaboost.adaboost_mrt import AdaboostMRT
from adaboost.error_functions import *
from typing import Tuple
from sklearn.exceptions import ConvergenceWarning,DataConversionWarning
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning) # ignore since we are artifically constraining convergence
warnings.filterwarnings("ignore", category=DataConversionWarning)


def generate_sample_line_data(min_x: float, max_x: float, n_items:int, noise_var:float)->Tuple[np.array, np.array]:
    x = np.random.uniform(min_x, max_x, n_items)
    y = x*5 + np.random.normal(0, noise_var, n_items)
    return x,y


if __name__=='__main__':
    random.seed(1234)
    x_train, y_train = generate_sample_line_data(-5,5,200, 2)
    x_test, y_test = generate_sample_line_data(-5,5,100, 0.0)

    # train Adaboost.MRT
    n_iterations = 5
    amrt = AdaboostMRT(base_learner=MLPRegressor, iterations=n_iterations)
    amrt.fit(x_train,y_train,N=100,phi=0.5,n=2, hidden_layer_sizes = (20,20), max_iter=4000, verbose=True, random_state=1234)

    # Apply to sample data
    for idx in range(0, n_iterations):
        y_test_predict = amrt.predict_individual(x_test, list(range(0,idx+1)))
        print(f"First {idx+1} learners, RMSE: {np.sqrt(((y_test_predict - y_test) ** 2).mean())}")

    y_test_predict = amrt.predict(x_test)
    print(f"Full Ensemble learners, RMSE: {np.sqrt(((y_test_predict - y_test) ** 2).mean())}")
    print('Complete')


    ## Compare to xgboost, NOTE: not yet tuned
    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.5,
                max_depth = 5, alpha = 50, n_estimators = n_iterations)
    xg_reg.fit(x_train.reshape(-1,1),y_train)
    preds = xg_reg.predict(x_test.reshape(-1,1))
    print(f"XGBoost, RMSE: {np.sqrt(((preds - y_test) ** 2).mean())}")
    print('Complete')
    
