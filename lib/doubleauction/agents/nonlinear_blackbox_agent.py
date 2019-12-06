from doubleauction.agents import MarketAgent
import numpy as np
from abc import abstractmethod


class NonlinearBlackBoxAgent(MarketAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        Linear blackbox agent. Only self_last_offer, current_time and previous_success are known
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


class NonlinearBlackBoxBuyer(NonlinearBlackBoxAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        A buyer who takes determines the new offer as a linear combination of all data available in observation
        """
        super().__init__(agent_id, reservation_price)

    def decide(self):
        
        if self.first_ever_bid:
            self.first_ever_bid = False
            
            demand = scipy.stats.halflogistic(-7.692926601910835e-08, 31.41266555783104).rvs()
        
            if self.reservation_price - demand < 0:
                demand = np.random.rand()*self.reservation_price

            return max(0, self.reservation_price - demand)
        
        
        if self.observations['previous_success']:
            new_offer = self.coefs[0]*self.reservation_price + self.coefs[1]*self.observations['self_last_offer']
        else:
            new_offer = (self.coefs[0] + self.coefs[2])*self.reservation_price + (self.coefs[1] - self.coefs[2])*self.observations['self_last_offer']
        if new_offer > self.reservation_price:
            return self.reservation_price
        else:
            return new_offer


class NonlinearBlackBoxSeller(NonlinearBlackBoxAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        A seller who takes determines the new offer as a linear combination of all data available in observation
        """
        super().__init__(agent_id, reservation_price)

    def decide(self):
        
        if self.first_ever_bid:
            self.first_ever_bid = False
            
            demand = scipy.stats.expon(0.0, 33.327542829759196).rvs()
            new_offer = self.reservation_price + demand
            if new_offer < self.reservation_price:
                return self.reservation_price
            else:
                return new_offer
        
        if self.observations['previous_success']:
            new_offer = self.coefs[0]*self.reservation_price + self.coefs[1]*self.observations['self_last_offer']
        else:
            new_offer = (self.coefs[0] + self.coefs[2])*self.reservation_price + (self.coefs[1] - self.coefs[2])*self.observations['self_last_offer']
        if new_offer < self.reservation_price:
            return self.reservation_price
        else:
            return new_offer
