import unittest
import numpy as np

from adaboost.adaboost_mrt import AdaboostMRT, weighted_sample
from unittest.mock import patch

class TestSampleLearner:
    def __init__(self) -> None:
        return None
        
    def fit(self, X, y):
        return True

    def predict(self, X):
        return np.array([1,1]).reshape(-1,1)

class TestAdaboostMrt((unittest.TestCase)):
    _sample_array1 = np.array([0,1])
    _sample_array2 = np.array([1,0])
    _sample_array3 = np.array([0,1,2])

    def test_weighted_sample_should_work_with_sample1(self):
        value = weighted_sample(self._sample_array1, 2, self._sample_array1)
        self.assertTrue(np.array_equal(value, np.array([1, 1])))

    def test_weighted_sample_should_work_with_sample2(self):
        value = weighted_sample(self._sample_array1, 2, self._sample_array2)
        self.assertTrue(np.array_equal(value, np.array([0, 0])))

    def test_weighted_sample_should_return_different_values(self):
        input_array = np.array(range(0, 100))
        value = weighted_sample(input_array, 10, np.ones(input_array.shape))
        self.assertTrue(len(np.unique(value)) > 1)

    def test_weighted_sample_should_return_correct_length(self):
        num_values:int = 10
        value = weighted_sample(self._sample_array1, num_values, self._sample_array2)
        self.assertEqual(len(value), num_values)

    def test_weighted_sample_should_raise_value_error_unequal_array(self):
        self.assertRaises(ValueError, weighted_sample, self._sample_array1, 2, self._sample_array3)


    def test_adaboost_mrt_should_work_with_fake_learner(self):
        abmrt = AdaboostMRT(base_learner=TestSampleLearner)
        abmrt.fit(X=[1, 2], y = [1,2], N=1)
        output = abmrt.predict([1])
        self.assertIsNotNone(output)

    