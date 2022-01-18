import math

import numpy as np
import data
import time
import tqdm

"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""


def sigmoid(a):
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying sigmoid (z from the slides).
    """
    return 1 / (1 + math.exp(-a))


def softmax(a):
    """
    Compute the softmax function.

    f(x) = (e^x) / Σ (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    exp
        the output of softmax
    """
    exp = np.exp(a)

    # Calculating softmax for all examples.
    for i in range(len(a)):
        exp[i] /= np.sum(exp[i])

    return exp


def binary_cross_entropy(y, t):
    """
    Compute binary cross entropy.

    L(x) = t*ln(y) + (1-t)*ln(1-y)

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        binary cross entropy loss value according to above definition
    """
    return (t * np.log(y) + (1 - t) * np.log(1 - y)).mean()


def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.

    L(x) = - Σ (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition
    """
    return -np.mean(np.log(y[np.arange(len(t)), t]))


class Network:
    def __init__(self, hyperparameters, activation, loss, out_dim):
        """
        Perform required setup for the network.

        Initialize the weight matrix, set the activation function, save hyperparameters.

        You may want to create arrays to save the loss values during training.

        Parameters
        ----------
        hyperparameters
            A dict contains hyper-parameters
        activation
            The non-linear activation function to use for the network
        loss
            The loss function to use while training and testing
        gradient
            The gradient function
        out_dim
            The output dimensions
        """
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss

        self.weights = np.zeros((28 * 28 + 1, out_dim))  # initialize weights to zeros

    def forward(self, X):
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for
        """
        return self.activation(np.matmul(X, self.weights))

    def __call__(self, X):
        return self.forward(X)

    def train(self, minibatch):
        """
        Train the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
        tuple containing:
            average loss over minibatch
            accuracy over minibatch
        """
        X, y = minibatch

        X = data.append_bias(X)  # append bias

        y_hat = self.forward(X)

        y_hot = data.onehot_encode(y)

        # gradient descendant
        grad = (1 / X.shape[0]) * np.dot(X.T, (y_hat - y_hot))
        self.weights = self.weights - self.hyperparameters["learning_rate"] * grad

        loss = self.loss(y_hat, y)

        preds = self.predict(X)
        acc = self.accuracy(preds, y)

        return loss, acc

    def test(self, minibatch):
        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
            tuple containing:
                average loss over minibatch
                accuracy over minibatch
        """
        X, y = minibatch

        X = data.append_bias(X)  # append bias

        y_hat = self.forward(X)

        loss = self.loss(y_hat, y)

        preds = self.predict(X)
        acc = self.accuracy(preds, y)

        return loss, acc

    def predict(self, X):
        y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)

    def accuracy(self, y, t):
        return np.sum(t == y) / len(t)
