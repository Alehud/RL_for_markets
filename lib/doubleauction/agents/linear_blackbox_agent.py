from doubleauction.agents import MarketAgent
import numpy as np
from abc import abstractmethod

import scipy.stats

class LinearBlackBoxAgent(MarketAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        Linear blackbox agent. The next offer is a linear combination of all obsevations agent has and his reservation price.
        He knows:
        'self_last_offer': previous offer of the agent
        'previous_success': whether the agent successfully made a deal in the previous round
        """
        super().__init__(agent_id, reservation_price)
        self.observations['previous_success'] = False
    
        self.first_ever_bid = True
        
    def new_game(self):
        self.first_ever_bid = True
        
    def new_round(self):
        pass

    def receive_observations_from_environment(self, env):
        agents = env.agents
        rewards = env.rewards

        self.reward += rewards[self.agent_id]

        agent_info = agents[agents['id'] == self.agent_id]
        self_last_offer = agent_info['last_offer'].iloc[0]
        previous_success = agent_info['previous_success'].iloc[0]

        self.observations['self_last_offer'] = self_last_offer
        self.observations['previous_success'] = previous_success

    @abstractmethod
    def decide(self, *args, **kwargs):
        return -1


class LinearBlackBoxBuyer(LinearBlackBoxAgent):
    def __init__(self, agent_id: str, reservation_price: float, noisy:bool):
        """
        Linear blackbox buyer. The next offer is a linear combination of all obsevations agent has and his reservation price.
        He knows:
        'self_last_offer': previous offer of the agent
        'previous_success': whether the agent successfully made a deal in the previous round
        """
        super().__init__(agent_id, reservation_price)
        
        self.noisy = noisy
        self.coeffs = np.array([ 0.75497335, 13.1468725 ])
        

    def compose_observation_vector(self):
        vals = np.array([self.reservation_price - self.observations['self_last_offer'], 
                         self.observations['previous_success']])
#         print(vals)
        
        return vals

    def decide(self):
        if self.first_ever_bid:
            self.first_ever_bid = False
            
            demand = scipy.stats.halflogistic(-7.692926601910835e-08, 31.41266555783104).rvs()
        
            if self.reservation_price - demand < 0:
                demand = np.random.rand()*self.reservation_price

            return max(0, self.reservation_price - demand)
        
        vals = self.compose_observation_vector()
        demand = np.dot(vals, self.coeffs)
        
        if self.noisy:
            demand += scipy.stats.laplace.rvs(loc=0,scale=.3)
            
        if demand < 0:
            demand = abs(scipy.stats.laplace.rvs(loc = 0, scale=2.))
            
#         print(demand)
        
        if demand > self.reservation_price:
            return abs(scipy.stats.laplace.rvs(loc =0, scale=2.))
        else:
            return self.reservation_price - demand


class LinearBlackBoxSeller(LinearBlackBoxAgent):
    def __init__(self, agent_id: str, reservation_price: float, noisy:bool):
        """
        Linear blackbox seller. The next offer is a linear combination of all obsevations agent has and his reservation price.
        He knows:
        'self_last_offer': previous offer of the agent
        'previous_success': whether the agent successfully made a deal in the previous round
        """
        super().__init__(agent_id, reservation_price)
        
        self.noisy = noisy
        self.coeffs = np.array([ 0.75497335, 13.1468725 ])

    def compose_observation_vector(self):
        vals = np.array([self.observations['self_last_offer'] - self.reservation_price,
                         self.observations['previous_success']])
        
#         print(vals)
        
        return vals

    def decide(self):
        
        if self.first_ever_bid:
            self.first_ever_bid = False
            
            demand = scipy.stats.expon(0.0, 33.327542829759196).rvs()
            new_offer = self.reservation_price + demand
            if new_offer < self.reservation_price:
                return self.reservation_price
            else:
                return new_offer
            
        vals = self.compose_observation_vector()
        demand = np.dot(vals, self.coeffs)
        
        if self.noisy:
            demand += scipy.stats.laplace.rvs(loc=0,scale=.3)
            
        if demand < 0:
            demand = abs(scipy.stats.laplace.rvs(loc = 0, scale=2.))
            
#         print(demand)
        
        return self.reservation_price + demand
        
