
from agents import Buyer, Seller
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
        
        
        
class RandomSeller(Seller):
    def __init__(self, agent_id: str,  reservation_price: float):
        """
        A seller agent that extends the market agent
        :param agent_id: a unique id that differentiates this agent to other agents
        :param reservation_price: the reservation price, or minimum price that this agent is
        willing to sell
        """
        super().__init__(agent_id, reservation_price)

        
    def decide(self, observations):
        demand = scipy.stats.expon(0.0, 33.327542829759196).rvs()
        
        return self.reservation_price + demand 
