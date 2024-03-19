import unittest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neural_network.activation import ActivationFunctions, BackwardActivation  # Replace with your actual package path

# test in the terminal : python -m unittest test_activation.py

class TestActivationFunctions(unittest.TestCase):

    def test_linear(self):
        """Test the linear activation function."""
        Z = np.array([-1, 0, 1])
        A_expected = Z
        A, cache = ActivationFunctions.linear(Z)
        self.assertTrue(np.array_equal(A, A_expected))
        self.assertTrue(np.array_equal(cache, Z))

    def test_sigmoid(self):
        """Test the sigmoid activation function."""
        Z = np.array([-1, 0, 1])
        A, cache = ActivationFunctions.sigmoid(Z)
        A_expected = 1 / (1 + np.exp(-Z))
        self.assertTrue(np.allclose(A, A_expected))
        self.assertTrue(np.array_equal(cache, Z))

    def test_relu(self):
        """Test the relu activation function."""
        Z = np.array([-1, 0, 1])
        A, cache = ActivationFunctions.relu(Z)
        A_expected = np.array([0, 0, 1])
        self.assertTrue(np.array_equal(A, A_expected))
        self.assertTrue(np.array_equal(cache, Z))

    def test_tanh(self):
        """Test the tanh activation function."""
        Z = np.array([-1, 0, 1])
        A, cache = ActivationFunctions.tanh(Z)
        A_expected = np.tanh(Z)
        self.assertTrue(np.allclose(A, A_expected))
        self.assertTrue(np.array_equal(cache, Z))

    def test_softmax(self):
        """Test the softmax activation function."""
        Z = np.array([[1, 2], [3, 4]])
        A, cache = ActivationFunctions.softmax(Z)
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        A_expected = expZ / np.sum(expZ, axis=0, keepdims=True)
        self.assertTrue(np.allclose(A, A_expected))
        self.assertTrue(np.array_equal(cache, Z))

class TestBackwardActivation(unittest.TestCase):

    def test_linear_activation_backward(self):
        dA = np.array([[-0.1, 0.2], [0.3, -0.4]])
        Z = np.array([[0.5, -0.2], [-0.3, 0.8]])
        expected_dZ = dA  # For linear activation, dZ should equal dA
        dZ = BackwardActivation.linear_activation_backward(dA, Z)
        self.assertTrue(np.array_equal(dZ, expected_dZ))

    def test_sigmoid_backward(self):
        dA = np.array([[0.1, -0.2], [0.3, -0.4]])
        Z = np.array([[0, 2], [-1, 0]])
        s = 1 / (1 + np.exp(-Z))
        expected_dZ = dA * s * (1 - s)
        dZ = BackwardActivation.sigmoid_backward(dA, Z)
        self.assertTrue(np.allclose(dZ, expected_dZ))

    def test_relu_backward(self):
        dA = np.array([[-0.1, 0.2], [0.3, -0.4]])
        Z = np.array([[0.5, -0.2], [-0.3, 0.8]])
        expected_dZ = np.array([[-0.1, 0], [0, -0.4]])  # For ReLU, dZ is zero where Z is <= 0
        dZ = BackwardActivation.relu_backward(dA, Z)
        self.assertTrue(np.array_equal(dZ, expected_dZ))

    def test_tanh_backward(self):
        dA = np.array([[0.1, -0.2], [0.3, -0.4]])
        Z = np.array([[0, 2], [-1, 0]])
        A = np.tanh(Z)
        expected_dZ = dA * (1 - np.power(A, 2))
        dZ = BackwardActivation.tanh_backward(dA, Z)
        self.assertTrue(np.allclose(dZ, expected_dZ))

    def test_softmax_backward(self):
        dA = np.array([[0.1, -0.2], [0.3, -0.4]])
        Z = np.array([[1, 2], [3, 4]])
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        A = expZ / np.sum(expZ, axis=0, keepdims=True)
        expected_dZ = dA * A * (1 - A)
        dZ = BackwardActivation.softmax_backward(dA, Z)
        self.assertTrue(np.allclose(dZ, expected_dZ))



if __name__ == '__main__':
    unittest.main()
