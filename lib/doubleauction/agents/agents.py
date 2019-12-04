__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"

from abc import abstractmethod
import numpy as np


class MarketAgent:
    def __init__(self, agent_id: str, reservation_price: float):
        """
        An market agent object. This class is extended to include all the agent logic for the
        agent interactions.
        :param agent_id: A unique id to differentiate this agent with other agents
        :param reservation_price: the reservation price, which is agents ideal price regarding a
        purchase or sale
        """
        self.agent_id = agent_id
        self.reservation_price = reservation_price
        self.coefs = None
        self.observations = {}
        self.reward = 0.0
        
    @abstractmethod
    def decide(self, *args, **kwargs):
        return -1

    @abstractmethod
    def receive_observations_from_environment(self, env):
        return -1


class Buyer(MarketAgent):
    def __init__(self, agent_id: str,  reservation_price: float):
        """
        A buyer agent that extends the market agent
        :param agent_id: a unique id that differentiates this agent to other agents
        :param reservation_price: the reservation price, or maximum price that this agent is
        willing to buy
        """
        super().__init__(agent_id, reservation_price)

    @abstractmethod
    def decide(self, *args, **kwargs):
        return -1
        
        
class Seller(MarketAgent):
    def __init__(self, agent_id: str,  reservation_price: float):
        """
        A seller agent that extends the market agent
        :param agent_id: a unique id that differentiates this agent to other agents
        :param reservation_price: the reservation price, or minimum price that this agent is
        willing to sell
        """
        super().__init__(agent_id, reservation_price)

    @abstractmethod
    def decide(self, *args, **kwargs):
        return -1
