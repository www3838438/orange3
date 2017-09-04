# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from Orange.data import Table
from Orange.modelling import NNLearner, ConstantLearner
from Orange.evaluation import CA, CrossValidation, MSE


class TestNNLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.housing = Table('housing')
        cls.learner = NNLearner()

    def test_NeuralNetwork(self):
        results = CrossValidation(self.iris, [self.learner], k=3)
        ca = CA(results)
        self.assertGreater(ca, 0.8)
        self.assertLess(ca, 0.99)

    def test_NeuralNetwork_regression(self):
        const = ConstantLearner()
        results = CrossValidation(self.housing, [self.learner, const], k=3)
        mse = MSE(results)
        self.assertLess(mse[0], mse[1])
