import scipy.stats
from doubleauction.agents import MarketAgent
from abc import abstractmethod


class RandomAgent(MarketAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        Random agent. His next offer is chosen randomly.
        """
        super().__init__(agent_id, reservation_price)

    def receive_observations_from_environment(self, env):
        rewards = env.rewards
        self.reward += rewards[self.agent_id]

    @abstractmethod
    def decide(self, *args, **kwargs):
        return -1


class RandomBuyer(RandomAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        Random buyer. His next offer is chosen randomly.
        """
        super().__init__(agent_id, reservation_price)

    def decide(self):
        demand = scipy.stats.halflogistic(-7.692926601910835e-08, 31.41266555783104).rvs()
        new_offer = self.reservation_price - demand
        if new_offer > self.reservation_price:
            return self.reservation_price
        else:
            return new_offer


class RandomSeller(RandomAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        Random seller. His next offer is chosen randomly.
        """
        super().__init__(agent_id, reservation_price)

    def decide(self):
        demand = scipy.stats.expon(0.0, 33.327542829759196).rvs()
        new_offer = self.reservation_price + demand
        if new_offer < self.reservation_price:
            return self.reservation_price
        else:
            return new_offer
