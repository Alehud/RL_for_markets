"""
Utility functions for the simulations
"""

import numpy as np


def generate_buyer_prices_paper(count, discrete=False):
    if discrete:
        return np.random.randint(low = 103, high = 148+1, size=count)
    else:
        return np.random.uniform(low = 103, high = 148, size=count)


def generate_seller_prices_paper(count, discrete=False):
    if discrete:
        return np.random.randint(low = 73, high = 118+1, size=count)
    else:
        return np.random.uniform(low = 73, high = 118, size=count)
    
def compute_cdf(values, start, end):
    values2 = np.sort(np.append(values, [start, end]))
    
    ii = np.arange(0, len(values)+1)
    ii = np.append(ii, [len(values)])
    
    return values2, ii
    

def plot_price_curves(buyers, sellers, range_prices=[70, 150]):
    import matplotlib.pyplot as plt
    
    vv, ii = compute_cdf(sellers, range_prices[0], range_prices[1])
    plt.step(vv, ii, where='post', label='Sellers')
    
    vv, ii = compute_cdf(buyers, range_prices[0], range_prices[1])
    plt.step(np.flip(vv), len(buyers) - np.flip(ii), where='pre', label='Buyers')
    
    
def compute_equilibrium_price(buyers, sellers):
    vv, ii = compute_cdf(sellers, 0, max(sellers.max(), buyers.max()) + 1)
    
    vv2, ii2 = compute_cdf(buyers, 0, max(sellers.max(), buyers.max()) + 1)
    vv2, ii2 = np.flip(vv2), len(buyers) - np.flip(ii2)
    
    if len(vv) > len(vv2):
        delta = len(vv) - len(vv2)
        vv2 = np.pad(vv2, (0,delta), 'edge')
        ii2 = np.pad(ii2, (0,delta), 'edge')
    elif len(vv2) > len(vv):
        delta = len(vv2) - len(vv)
        vv = np.pad(vv, (0,delta), 'edge')
        ii = np.pad(ii, (0,delta), 'edge')
    
    test = (vv2 <= vv)
    if not np.any(test):
        return None
    else:
        d = np.argmax(test)
        
        return (max(vv[d-1], vv2[d]), min(vv2[d-1], vv[d]))
    