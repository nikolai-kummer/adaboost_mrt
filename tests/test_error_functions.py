import unittest

import numpy as np

from adaboost.error_functions import * 

class TestErrorFunctions((unittest.TestCase)):

    test_array1 = np.array([0,1])
    test_array2 = np.array([1,3])
    test_array3 = np.array([0,1, 2])


    # variance scaled error
    def test_variance_scaled_error_should_raise_value_error_unequal_shape(self):
        self.assertRaises(ValueError, variance_scaled_error, self.test_array1, self.test_array3)

    def test_variance_scaled_error_should_raise_zero_division_error_for_zero_prediction_error(self):
        self.assertRaises(ZeroDivisionError, variance_scaled_error, self.test_array1, self.test_array1)

    def test_variance_scaled_error_should_work_with_sample(self):
        value = variance_scaled_error(self.test_array1, self.test_array2)
        self.assertEqual(value, 0.0)



    # absolute relative error
    def test_absolute_relative_error_should_raise_value_error_unequal_shape(self):
        self.assertRaises(ValueError, variance_scaled_error, self.test_array1, self.test_array3)

    def test_absolute_relative_error_should_raise_value_error_zero_in_true_value(self):
        self.assertRaises(ZeroDivisionError, variance_scaled_error, self.test_array1, self.test_array1)

    def test_variance_scaled_error_should_work_with_sample(self):
        value = variance_scaled_error(self.test_array1, self.test_array2)
        self.assertTrue(np.array_equal(value, np.array([2.0, 4.0])))