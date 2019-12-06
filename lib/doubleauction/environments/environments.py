__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"

from gym import Env
from abc import abstractmethod
import pandas as pd
import numpy as np
from gym.spaces import Box


class MarketEnvironment(Env):
    def __init__(self, sellers: np.ndarray, buyers: np.ndarray, max_time: int, matcher):
        """
        An abstract market environment extending the typical gym environment
        :param sellers: A list containing all the agents that are extending the Seller agent
        :param buyers: A list containing all the agents that are extending the Buyer agent
        :param max_time: the maximum number of steps that runs for this round.
        """
        self.n_sellers = np.size(sellers)
        self.n_buyers = np.size(buyers)
        self.max_time = max_time
        self.matcher = matcher
        self.agents = pd.DataFrame([dict(id=x.agent_id, res_price=x.reservation_price, role='Seller', done=None, last_offer=None, stop_time=None, previous_success=False) for x in sellers] +
                                   [dict(id=x.agent_id, res_price=x.reservation_price, role='Buyer', done=None, last_offer=None, stop_time=None, previous_success=False) for x in buyers])
        self.not_done_sellers = None
        self.not_done_buyers = None
        self.deal_history: list = None
        self.rewards: dict = None
        self.time = None
        self.if_round_done = None

        sellers_res_prices = self.agents[self.agents['role'] == 'Seller']['res_price']
        buyers_res_prices = self.agents[self.agents['role'] == 'Buyer']['res_price']
        self.action_space = Box(
            np.concatenate((np.array(sellers_res_prices, dtype=np.float32), np.array([0.0] * self.n_buyers, dtype=np.float32))),
            np.concatenate((np.array([np.infty] * self.n_sellers, dtype=np.float32), np.array(buyers_res_prices, dtype=np.float32)))
        )
        self.reset()
        
        

    def reset(self):
        """
        Resets the environment to an initial state, so that the game can be repeated.
        """
        self.deal_history = list()
        self.time = 0
        self.if_round_done = False
        self.agents['done'] = False
#         self.agents['last_offer'] = 0.0
        self.not_done_sellers = np.array([False] * self.n_sellers)
        self.not_done_buyers = np.array([False] * self.n_buyers)
        # These are current rewards in the round, not the cumulative rewards of agents
        self.rewards = {agent_id: 0 for agent_id in self.agents['id']}
        
        self.first_in_round = True

    def step(self, current_offers):
        """
        The step function takes the agents' new offers and simulates one time step in the market
        :param current_offers: a dictionary containing the offer per agent
        """
        
        if self.first_in_round:
            self.first_in_round = False
            
            self.agents['previous_success'] = False
            
#             print(self.agents['previous_success'])
        
        # self.deal_history and self.agents are also updated in match()
        self.rewards = self.matcher.match(
            current_offers=current_offers,
            env_time=self.time,
            agents=self.agents,
            deal_history=self.deal_history
        )
        # observations = dict((agent_id, self.get_observation_state(agent_id=agent_id)) for agent_id in self.agents['id'])
        self.time += 1

        # Updating masks (arrays of booleans) for agents who are not done yet in this round
        self.not_done_sellers = ~np.array(self.agents[self.agents['role'] == 'Seller']['done'])
        self.not_done_buyers = ~np.array(self.agents[self.agents['role'] == 'Buyer']['done'])
        not_done_agents = self.agents[~self.agents['done']]

        # Checking if the round terminated
        if (self.time == self.max_time) or \
                not (True in list(self.not_done_buyers)) or not (True in list(self.not_done_sellers)) or \
                not_done_agents[not_done_agents['role'] == 'Seller']['res_price'].min() > not_done_agents[not_done_agents['role'] == 'Buyer']['res_price'].max():
            self.if_round_done = True
            self.agents.loc[self.agents['done'] == False, 'previous_success'] = False
            self.agents.loc[self.agents['done'] == True, 'previous_success'] = True

        # Determining action space
        # temp = self.agents[['role', 'res_price', 'done']]
        # temp.loc[temp['done'] == True, 'res_price'] = -1
        # temp.loc[temp['role'] == 'Buyer', 'res_price'] = 0
        # low_actions = list(temp[temp['role'] == 'Seller']['res_price']) + list(temp[temp['role'] == 'Buyer']['res_price'])
        # temp = self.agents[['role', 'res_price', 'done']]
        # temp.loc[temp['done'] == True, 'res_price'] = -1
        # temp.loc[temp['role'] == 'Seller', 'res_price'] = np.inf
        # high_actions = list(temp[temp['role'] == 'Seller']['res_price']) + list(temp[temp['role'] == 'Buyer']['res_price'])
        #
        # self.action_space = Box(np.array(low_actions), np.array(high_actions))

    @abstractmethod
    def render(self, mode='human'):
        """
        This method renders the environment in a specific visualiation. e.g. human is to render
        for a human observer.
        :param mode: Please check the gym env docstring
        :return: A render object
        """
        pass
