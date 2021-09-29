from typing import Callable, List, Union
import numpy as np
import random

from adaboost.error_functions import * 



def weighted_sample(input_array: np.array, n_items: int, weights: np.array) -> List:
    """takes an input list and returns a weighted sample

    Args:
        input_array (List): array to sample from
        n_items (int): number of items to get
        weights (List): weights for each element in list

    Raises:
        ValueError: when the input_array and weights are not the same shape

    Returns:
        List: returns sampled array of indexes
    """
    # Check for equal items
    if input_array.shape != weights.shape:
        raise ValueError("Sampling array and weoght array must be same shape")
    return random.choices(input_array, weights,k= n_items)




class AdaboostMRT:
    # Implement for single dimensional output
    # Initial Parameters
    _base_learner = None  # base learner object
    n_iterations: int = 10  # Parameter T from original paper
    _phi: np.array = None  # Threshold parameter phi from original paper

    n_iteration: int = 0  # t parameter from original paper
    m: int = None  # Number of input samples
    n: float = None  # power by which to raise the error
    D_t: np.array = None  # sampling weight distribution for input characters
    D_r: np.array = None  # output weight distribution
    n_samples: int = None  # number of input samples
    n_input_features: int = None # number of input features in x
    _learner_array: List = []  # list of base_learner arrays

    def __init__(self, base_learner: object, iterations: int = 10):
        """[summary]

        Args:
            base_learner ([type]): base learner object that has a 
            iterations (int, optional): [description]. Defaults to 10.
        """
        # Check for base_learner properties
        self._base_learner = base_learner
        self.n_iterations = iterations

    def reshape_input(self, input_x:Union[np.array,List])->np.array:
        """Common function to reshpe X array into common shape for training

        Args:
            input_x (Union[np.array,List]): input array of shape [n_samples, n_features] or List (for n_features=1)

        Returns:
            np.array: output formatterd according to shape of (n_samples, n_features)
        """
        # Error checking on X input
        if isinstance(input_x, list):
            input_x = np.array(input_x).reshape(-1,1)
        elif isinstance(input_x, np.ndarray):
            if input_x.ndim == 1:   # check for singleton
                input_x=input_x.reshape(-1,1)

        return input_x




    def fit(self, X: np.array, y: np.array, N:int,  phi:np.array = 0.1, n:float = 1, error_function: Callable=variance_scaled_error, verbose:bool = False, **kwargs):
        """
            Accepts input vector X of size (m,p) and output vector of size (m,R) and iteratively runs
            the adaboost.mrt algorithm

            Parameters:
                X (np.array): input array with m examples (columns) 
                y (np.array): output array of shape mxR
                N (int): number of items to sample from the data 
                n (float): error power (1 for linear error, 2 for squared error, etc)
                phi (np.array): (Optional) Weighting array of shape (1,R), initialized to 0.6 for all
                verbose (bool): (Optional, default False) whether to print out messages  
                kwargs: optional parameters for the learners
        """
        # Error checking on X input
        X = self.reshape_input(X)

        self.n_iteration = 0 
        self.m = X.shape[0]
        self.n_samples = X.shape[0]
        self.n_input_features = X.shape[1]
        self.N = N  # Number of items toasample from the data at each iteration
        self.n = n  # error power (raises the error to this power, 1 for linear, 2 for quadratic)
        self._num_outputs = 1 #y.shape[1], r parameter in paper

        

        self.D_t = np.ones((1,self.n_samples))/self.n_samples # sampling weight distribution
        self.D_y = np.ones((1,self.n_samples))/self.n_samples # output error distribution
        self.epsilon = np.zeros((self._num_outputs ,self.n_iterations))  # misclassification error rate
        self.beta_t = np.zeros((self._num_outputs ,self.n_iterations))  # weight updating parameter


        for t in range(0,self.n_iterations):
            #TODO:  error check the D_t
            sample_idx = np.sort(weighted_sample(np.array(range(0, self.m)), self.N, self.D_t[0,]))
            if verbose:
                print(f'Unique samples indices: {len(np.unique(sample_idx))} out of {len(sample_idx)} from total data: {self.m}')

            # initialize and train learner
            sample_learner = self._base_learner(**kwargs)
            sample_learner.fit(X[sample_idx], y[sample_idx])
            self._learner_array.append(sample_learner)

            # Calculate errors
            y_predict = sample_learner.predict(X)
            error_measure = absolute_relative_error(y_predict, y) # absolute relative error
            ind_m = error_measure > phi

            # Calculate the misclassigication error rate for every output variable
            self.epsilon[:,t] = np.sum(self.D_y*ind_m)

            # Set the weight updating parameter
            self.beta_t[:,t] = self.epsilon[:,t]**self.n

            # Update the output error distribution
            temp_b = np.ones(ind_m.shape)
            temp_b[np.bitwise_not(ind_m)] = temp_b[np.bitwise_not(ind_m)] * self.beta_t[:,t]
            self.D_y = self.D_y * temp_b
            self.D_y = self.D_y * 1/np.sum(self.D_y)
            self.D_t = np.mean(self.D_y, axis=0).reshape(-1,self.m)


    def predict(self, X:np.array) -> np.array:
        """Run ensemble prediction on input vector

        Args:
            X (np.array): input vector

        Returns:
            np.array: averaged ouput of the ensemble
        """
        # Error checking on X input
        X = self.reshape_input(X)

        output = np.zeros(self._num_outputs,)
        for predictor in self._learner_array:
            output = output + predictor.predict(X)

        return output/self.n_iterations
            

    def predict_individual(self, X:np.array, learner_index: Union[int, List]) -> np.array:
        """predicts individual learner or a list of the actual learners 

        Args:
            X (np.array): input vector
            learner_index (Union[int, List]): index of learners to use

        Raises:
            IndexError: raised if index is outside num_iteration or 0

        Returns:
            np.array: outputs prediction
        """
        if isinstance(learner_index, int):
            learner_index = [learner_index]
        if max(learner_index) > self.n_iterations or min(learner_index) < 0 or len(learner_index)==0:
            raise IndexError(f"Passed Index is out of bounds of [0, {self.n_iterations}]. Received: [{learner_index}]")

        # Error checking on X input
        X = self.reshape_input(X)

        output = np.zeros(self._num_outputs,)
        learner_array = [self._learner_array[idx] for idx in learner_index] 
        for predictor in learner_array:
            output = output + predictor.predict(X)

        return output/len(learner_index)
            

