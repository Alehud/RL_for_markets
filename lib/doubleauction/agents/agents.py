__author__ = "Thomas Asikis, Batuhan Yardim, Aleksei Khudorozhkov, Ka Rin Sim, Neri Passaleva"
__credits__ = ["Copyright (c) 2019 Thomas Asikis, Batuhan Yardim, Aleksei Khudorozhkov, Ka Rin Sim, Neri Passaleva"]
__license__ = "MIT"
__version__ = "0.2"
__maintainer__ = "Thomas Asikis, Batuhan Yardim, Aleksei Khudorozhkov, Ka Rin Sim, Neri Passaleva"

from abc import abstractmethod


class MarketAgent:
    def __init__(self, agent_id: str, reservation_price: float):
        """
        A market agent object. This class is extended to include all the agent logic for the
        agent interactions.
        :param agent_id: A unique id (string) of the agent
        :param reservation_price: the reservation price, the limit beyond which the agent would not make an offer
        """
        self.agent_id = agent_id
        self.reservation_price = reservation_price
        # array of coefficients, which are adjusted during the learning procedure
        self.coefs = None
        # observations of the agent
        self.observations = {}
        # cumuative reward
        self.reward = 0.0
        
    @abstractmethod
    def decide(self, *args, **kwargs):
        """
        Agent decides on the next offer
        :return: new offer
        """
        return -1

    @abstractmethod
    def receive_observations_from_environment(self, env):
        """
        Function for receiving observations from environment
        :param env: market environment
        :return: no return
        """
        return -1


class Buyer(MarketAgent):
    def __init__(self, agent_id: str,  reservation_price: float):
        """
        A buyer agent.
        :param agent_id: A unique id (string) of the agent
        :param reservation_price: the budget of the buyer, he/she cannot offer more
        """
        super().__init__(agent_id, reservation_price)

    @abstractmethod
    def decide(self, *args, **kwargs):
        return -1
        
        
class Seller(MarketAgent):
    def __init__(self, agent_id: str,  reservation_price: float):
        """
        A seller agent.
        :param agent_id: A unique id (string) of the agent
        :param reservation_price: the production cost of the seller, he/she cannot offer for less price
        """
        super().__init__(agent_id, reservation_price)

    @abstractmethod
    def decide(self, *args, **kwargs):
        return -1
