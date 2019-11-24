import pandas as pd
import numpy as np
from agents import Buyer, Seller
from environments import MarketEnvironment
from info_settings import InformationSetting
from matchers import RandomMatcher


def use_brain(observation, agent_id, environment, coefs):
    current_time = environment.time
    max_time = environment.max_steps
    n_sellers = environment.n_sellers
    n_buyers = environment.n_buyers

    values = np.array([])
    for val in observation.values():
        values = np.append(values, val)
    coefs = np.zeros_like(values)
