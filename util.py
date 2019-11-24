"""
Utility functions for the simulations
"""

import numpy as np


def generate_buyer_prices_paper(count):
    return np.random.uniform(low = 103, high = 148, size=count)

    
def generate_seller_prices_paper(count):
    return np.random.uniform(low = 73, high = 118, size=count)
    
