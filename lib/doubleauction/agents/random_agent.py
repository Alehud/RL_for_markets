
from .agents import Buyer, Seller
import scipy.stats
import numpy as np


class RandomBuyer(Buyer):
    def __init__(self, agent_id: str,  reservation_price: float):
        """
        A random buyer.
        """
        super().__init__(agent_id, reservation_price)
        
    def decide(self, observations):
        demand = scipy.stats.halflogistic(-7.692926601910835e-08, 31.41266555783104).rvs()
        return np.abs(self.reservation_price - demand)

    def receive_observations_from_environment(self, observations):
        self.observations['self_last_offer'] = observations['self_last_offer']
        self.observations['current_time'] = observations['current_time']
        

class RandomSeller(Seller):
    def __init__(self, agent_id: str,  reservation_price: float):
        """
        A random seller
        """
        super().__init__(agent_id, reservation_price)

        
    def decide(self, observations):
        demand = scipy.stats.expon(0.0, 33.327542829759196).rvs()
        return self.reservation_price + demand

    def receive_observations_from_environment(self, observations):
        self.observations['self_last_offer'] = observations['self_last_offer']
        self.observations['current_time'] = observations['current_time']
