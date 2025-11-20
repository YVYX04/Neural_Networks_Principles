                    # +––––––––––––––––+
                    # |   Network.py   |
                    # +––––––––––––––––+
                    # Yvan Richard - Winter 2025

                    # description:
                    # This module defines the Network class for building and training neural networks.
                    # It is largely inspired by Michael Nielsen's implementation in his book
                    # "Neural Networks and Deep Learning" (as referenced in my repository).


import numpy as np
import utils

class Network:
    """
    A simple neural network class inspired by Michael Nielsen's implementation.
    """
    # constructor: take numpy array as input for the layers' size
    def __init__(self, sizes):
        self.num_layers_ = len(sizes)
        self.sizes_ = sizes

        # randomly generated biases (from the standard Gaussian distribution)
        # bias generated for the hidden layers and output layer
        self.bias_ = [np.random.randn(y, 1) for y in sizes[1:]] 

        # generate the weights
        self.weights_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # feedforward method
    def feedforward(self, a):
        """
        In this function, we propagate the weighted sum to the next layer a_next.
        Subsequently, we activate it with the sigmoid function.
        """

        for W, b in zip(self.weights_, self.bias_):
            a_next = W @ a + b
            a_next = utils.sigmoid(a_next)

        return a_next


        