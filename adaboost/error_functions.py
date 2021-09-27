import numpy as np

def absolute_relative_error(prediction: np.array, true_value: np.array):
    """
    Returns an estimate of the error Er_t(i) in the form abs( (f_t(x_i) - y_i) / y_i).
    This is the original formulation of Adaboost.RT
    Args:
        prediction (np.array): [description]
        true_value (np.array): [description]

    Raises:
        ValueError -- Checks if the input and output have the same shape
        ZeroDivisionError -- Checks if true_value contains zero

    Returns:
        [type]: [description]
    """
    if prediction.shape != true_value.shape:
        raise ValueError("Prediction and true value array must be same shape")
    elif np.any(true_value==0):
        raise ZeroDivisionError("True value cannot contain zeros in the true values. Use variance_scaled_error function instead")
    
    return np.abs(np.divide(prediction - true_value,true_value) )

def variance_scaled_error(prediction: np.array, true_value: np.array):
    """[summary]

    Args:
        prediction (np.array): [description]
        true_value (np.array): [description]

    Raises:
        ValueError -- Checks if the input and output have the same shape
        ZeroDivisionError -- Raised if there is no variance in the true value

    Returns:
        [type]: [description]
    """
    if prediction.shape != true_value.shape:
        raise ValueError("Prediction and true value array must be same shape")

    absolute_error = np.abs(prediction - true_value)
    absolute_error_std_dev = np.sqrt(np.var(absolute_error))
    
    if absolute_error_std_dev == 0:
        raise ZeroDivisionError("Prediction error does not have any variance")
    
    return absolute_error/absolute_error_std_dev
