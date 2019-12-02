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
        :return: the initial state so that a new round begins.
        """
        self.deal_history = list()
        self.time = 0
        self.if_round_done = False
        self.agents['done'] = False
        self.agents['last_offer'] = 0.0
        self.not_done_sellers = np.array([False] * self.n_sellers)
        self.not_done_buyers = np.array([False] * self.n_buyers)

    def get_observation_state(self, agent_id: str):
        """
        The method that generates the state for the agents, based on the information setting.
        :param agent_id: usually a string, a unique id for an agent
        :return: the observation space
        """
        agent_info = self.agents[self.agents['id'] == agent_id]
        agent_role = agent_info['role'].iloc[0]
        same_side_agents = self.agents[self.agents['role'] == agent_role]
        other_side_agents = self.agents[self.agents['role'] != agent_role]
        self_last_offer = agent_info['last_offer'].iloc[0]
        same_side_last_offers = np.array(same_side_agents['last_offer'])
        same_side_res_prices = np.array(same_side_agents['res_price'])
        same_side_not_done = len(same_side_agents.loc[same_side_agents['done'] == False])
        other_side_last_offers = np.array(other_side_agents['last_offer'])
        other_side_res_prices = np.array(other_side_agents['res_price'])
        other_side_not_done = len(other_side_agents.loc[other_side_agents['done'] == False])
        completed_deals = [(x['time'], x['deal_price']) for x in self.deal_history]
        previous_success = agent_info['previous_success'].iloc[0]

        observation_state = {
            'self_last_offer': self_last_offer,
            'same_side_last_offers': same_side_last_offers,
            'same_side_res_prices': same_side_res_prices,
            'same_side_not_done': same_side_not_done,
            'other_side_last_offers': other_side_last_offers,
            'other_side_res_prices': other_side_res_prices,
            'other_side_not_done': other_side_not_done,
            'completed_deals': completed_deals,
            'current_time': self.time,
            'max_time': self.max_time,
            'n_sellers': self.n_sellers,
            'n_buyers': self.n_buyers,
            'previous_success': previous_success
        }
        return observation_state

    def step(self, current_offers):
        """
        The step function takes the agents actions and returns the new state, reward,
        if the state is terminal and any other info.
        :param current_offers: a dictionary containing the offer per agent
        :return: a tuple of 4 objects: the object describing the next state, a data structure
        containing the reward per agent, a data structure containing boolean values expressing
        whether an agent reached a terminal state
        """
        rewards = self.matcher.match(
            current_offers=current_offers,
            env_time=self.time,
            agents=self.agents,
            deal_history=self.deal_history
        )
        observations = dict((agent_id, self.get_observation_state(agent_id=agent_id)) for agent_id in self.agents['id'])
        self.time += 1

        # Updating masks (arrays of booleans) for agents who are not done yet in this round
        self.not_done_sellers = ~np.array(self.agents[self.agents['role'] == 'Seller']['done'])
        self.not_done_buyers = ~np.array(self.agents[self.agents['role'] == 'Buyer']['done'])

        # Checking if the round terminated
        if (self.time == self.max_time) or \
                not (False in list(self.agents[self.agents['role'] == 'Seller']['done'])) or \
                not (False in list(self.agents[self.agents['role'] == 'Buyer']['done'])):
            self.if_round_done = True
            self.agents[self.agents['done'] == False]['previous_success'] = False
            self.agents[self.agents['done'] == True]['previous_success'] = True

        # Determining action space
        temp = self.agents[['role', 'res_price', 'done']]
        temp.loc[temp['done'] == True, 'res_price'] = -1
        temp.loc[temp['role'] == 'Buyer', 'res_price'] = 0
        low_actions = list(temp[temp['role'] == 'Seller']['res_price']) + list(temp[temp['role'] == 'Buyer']['res_price'])
        temp = self.agents[['role', 'res_price', 'done']]
        temp.loc[temp['done'] == True, 'res_price'] = -1
        temp.loc[temp['role'] == 'Seller', 'res_price'] = np.inf
        high_actions = list(temp[temp['role'] == 'Seller']['res_price']) + list(temp[temp['role'] == 'Buyer']['res_price'])

        self.action_space = Box(np.array(low_actions), np.array(high_actions))

        return observations, rewards, self.if_round_done

    @abstractmethod
    def render(self, mode='human'):
        """
        This method renders the environment in a specific visualiation. e.g. human is to render
        for a human observer.
        :param mode: Please check the gym env docstring
        :return: A render object
        """
        pass
