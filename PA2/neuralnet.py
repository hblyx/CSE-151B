################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################

import numpy as np
import data
import math


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta)
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError("%s is not implemented." % (activation_type))

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.a = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        self.a = a

        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        return x * (x > 0)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.a) * (1.0 - self.sigmoid(self.a))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return (1 - np.tanh(self.a) ** 2)

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        x = self.a
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


class Layer:
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(1024, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = math.sqrt(2 / in_units) * np.random.randn(in_units,
                                                           out_units)  # You can experiment with initialization.
        self.b = np.zeros((1, out_units))  # Create a placeholder for Bias
        self.x = None  # Save the input to forward in this
        self.a = None  # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x

        self.a = np.dot(x, self.w) + self.b

        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        self.d_x = np.dot(delta, self.w.T)

        self.d_w = np.dot(self.x.T, delta)

        self.d_b = np.sum(delta, axis=0)

        return self.d_x


class NeuralNetwork:
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        # save x
        self.x = x
        self.targets = targets

        # go through all layers forward
        z = x
        for i in range(len(self.layers)):
            z = self.layers[i](z)  # use the output of previous layer as the input

        self.y = self.softmax(z)  # for the last layer, we use softmax instead of activation

        # loss and target
        if self.targets is not None:
            loss = self.loss(self.y, self.targets)

            return self.y, loss

        return self.y

    def backward(self):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """
        # edge case (no targets specified)
        if self.targets == None:
            raise AssertionError("Targets not specified (targets == None)")

        delta = self.y - data.one_hot_encoding(self.targets)  # delta for the output layer: delta_k

        for i in range(len(self.layers) - 1, -1, -1):  # go through layers except the first layer
            # pass delta
            delta = self.layers[i].backward(delta)  # after loop, it gets the final delta

    def softmax(self, x):
        """
        Implement the softmax function here.
        Remember to take care of the overflow condition.
        """
        exp = np.exp(x)

        return exp / np.sum(exp, axis=1, keepdims=True)

    def loss(self, logits, targets):
        """
        Compute the categorical cross-entropy loss and return it.
        """
        return -np.mean(np.log(logits[np.arange(len(targets)), targets]))

    def predict(self):
        return np.argmax(self.y, axis=1)

    def accuracy(self):
        if self.targets == None:
            raise AssertionError("Targets not specified (targets == None)")
        if self.y == None:
            raise AssertionError("y has not been calculated (y == None)")

        return np.sum(self.targets == self.predict()) / len(self.targets)