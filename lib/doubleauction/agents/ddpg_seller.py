from .agents import Buyer, Seller
import scipy.stats
import numpy as np
from ..models import Actor
import torch
        
class DDPGSeller(Seller):
    def __init__(self, agent_id: str,  reservation_price: float, model:Actor, 
                         std_noise=50., anneal_steps=1e4, min_noise=2.):
        """
        A seller agent that extends the market agent
        :param agent_id: a unique id that differentiates this agent to other agents
        :param reservation_price: the reservation price, or minimum price that this agent is
        willing to sell
        """
        super().__init__(agent_id, reservation_price)
        self.std_noise = std_noise
        self.anneal_steps = anneal_steps
        self.min_noise = min_noise
        
        self.noise_step = (std_noise - min_noise) / anneal_steps
        
        self.sigma = std_noise
        
        self.last_observation = None
        self.last_action = None
        self.eval = False
        self.last_offer = 0
        
        
    def reset_round(self):
        self.last_offer = 0
        self.last_observation = None
        self.last_action = None
        
        
    def decide(self, observations):
        pass
#         s = torch.tensor(self.last_offer )
        
#         if self.eval:
            
#         else:
#             self.sigma -= self.noise_step
#             n = np.random.normal(0, self.sigma)

#             prev_

#             demand = model()
        
#         return self.reservation_price + demand 
    
    

