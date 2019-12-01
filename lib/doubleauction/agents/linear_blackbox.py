
from .agents import Buyer, Seller
import scipy.stats
import numpy as np

class LinearBlackboxBuyer(Buyer):
    def __init__(self, agent_id: str,  reservation_price: float):
        """
        A buyer.
        """
        super().__init__(agent_id, reservation_price)
        
    def decide(self, observations):
        demand = scipy.stats.halflogistic(-7.692926601910835e-08, 31.41266555783104).rvs()
        
        return np.abs(self.reservation_price - demand)
        
        
        
class LinearBlackboxSeller(Seller):
    def __init__(self, agent_id: str,  reservation_price: float):
        """
        A seller agent 
        """
        super().__init__(agent_id, reservation_price)

        
    def decide(self, observations):
        demand = scipy.stats.expon(0.0, 33.327542829759196).rvs()
        
        return self.reservation_price + demand 

