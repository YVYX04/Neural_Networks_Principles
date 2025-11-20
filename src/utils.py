                    # +––––––––––––––––+
                    # |   utils.py   |
                    # +––––––––––––––––+
                    # © 2025 Yvan Richard
                    # Winter 2025

# description:
# This module contains utility functions for neural network operations.

import numpy as np

def sigmoid(z):
    """
    The sigmoid activation function.
    """
    return 1.0 / (1.0 + np.exp(-z))

