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
import random as rd

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
 
        # generate the weights (also init from Gaussian standard distribution)
        # the dimension of each generated matrix is given by the dimension of the
        # two connected layers. As we can see, the list comprehension starts at:
        # ((input layer dim) x (first hidden layer dim))
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
    
    # Stochastic Gradient Descent method
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        """
        n = len(training_data)
        if test_data:
            n_test = len(test_data)

        # shuffle the training data
        rd.shuffle(training_data)

        # split the data in different mini bacthes
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

        # update the paramter theta based on the mini batches
        self.update_mini_bacthes(mini_batches, eta) # eta still the learning rate

    # update paramters based on the mini batches
    def update_mini_batches(self, mini_batch, eta):
        """
        Update the parameter theta with gradient descent based on
        a given learning rate and a given set of mini batches
        """

        # init the gradients to 0
        # vectors of 0
        nabla_b = [np.zeros(b.shape) for b in self.bias_]

        # matrices of 0
        nabla_W = [np.zeros(W.shape) for W in self.weights_]

        # backpropagation algorithm
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_W = [nw + dnw for nw, dnw in zip(nabla_W, delta_nabla_w)]

        # update the parameters
        self.weights_ = [w - (eta/len(mini_batch)) * nw 
                        for w, nw in zip(self.weights_, nabla_W)]
        
        self.bias_ = [b - (eta/len(mini_batch)) * nb 
                       for b, nb in zip(self.bias_, nabla_b)]
        
    # backpropagation algorithm
    def backprop(self, x, y):
        """
        to be implemented
        """
        return (x, y)
    
    # other functions




    

        


        