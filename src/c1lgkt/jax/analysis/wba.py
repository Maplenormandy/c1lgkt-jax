"""
Module containing helper functions for weighted Birkhoff averages. Generally only uses numpy
"""

import numpy as np
from jaxtyping import Real, ArrayLike

# %%

def wba_weights(n: int) -> Real[ArrayLike, "{n}"]:
    """
    Computes the WBA weights for an interval of length n
    
    :param n: Length of WBA window
    :type n: int
    :return: Normalized weights
    :rtype: ArrayLike
    """

    s = np.arange(n) / n
    weights = np.empty_like(s)
    weights[0] = 0
    weights[1:] = np.exp(-1.0 / (s[1:] * (1.0 - s[1:])))
    weights = weights / np.sum(weights)

    return weights
