__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"

import random
import pandas as pd
import numpy as np
from abc import abstractmethod


class Matcher:
    def __init__(self):
        """
        Abstract matcher object. This object is used by the Market environment to match agent offers
        and also decide the deal price.
        """
        pass

    @abstractmethod
    def match(self, current_offers: dict, env_time: int, agents: pd.DataFrame, deal_history: pd.DataFrame):
        """
        The matching method, which relies on several data structures passed from the market object.
        :param current_offers: A dictionary of agent id and offer value
        :param env_time: the current time step in the market
        :param agents: the dataframe containing the agent information
        :param deal_history: the dictionary containing all the successful deals till now
        :return: the dictionary containing the agents' ids as keys and the rewards as values
        """
        rewards: dict = None
        return rewards


class RandomMatcher(Matcher):
    def __init__(self, reward_on_reference=False):
        """
        A random matcher, which decides the deal price of a matched pair by sampling a uniform
        distribution bounded in [seller_ask, buyer_bid] range.
        The reward is calculated as the difference from cost or the difference to budget for
        sellers and buyers.
        :param reward_on_reference: The parameter to use a different reward calculation.
        If set to true the reward now becomes: offer - reservation price for, sellers
        and: reservation price - offer, for buyers.
        You may chose to use this reward scheme, but you have to justify why it is better than
        the old!
        """
        super().__init__()
        self.reward_on_reference = reward_on_reference

    def match(self, current_offers: dict, env_time: int, agents: pd.DataFrame, deal_history: pd.DataFrame):
        """
        The matching method, which relies on several data structures passed from the market object.
        :param current_offers: A dictionary of agent id and offer value
        :param env_time: the current time step in the market
        :param agents: the dataframe containing the agent information
        :param deal_history: the dictionary containing all the successful deals till now
        :return: the dictionary containing the agents' ids as keys and the rewards as values
        """
        # Update offers in environment class
        for agent_id, offer in current_offers.items():
            if not agents[agents['id'] == agent_id]['done'].iloc[0]:
                agents.loc[agents['id'] == agent_id, 'last_offer'] = offer

        # Keep buyer and seller offers with non-matched ids sorted:
        # descending by offer value for buyers
        # ascending by offer value for sellers
        # and do a second sorting on ascending id to break ties for both
        buyers_sorted = agents[(agents['role'] == 'Buyer') &
                              (agents['done'] == False)].sort_values(['last_offer', 'id'], ascending=[False, True])

        sellers_sorted = agents[(agents['role'] == 'Seller') &
                               (agents['done'] == False)].sort_values(['last_offer', 'id'], ascending=[True, True])

        min_len = min(sellers_sorted.shape[0], buyers_sorted.shape[0])
        rewards = dict((a_id, 0) for a_id in agents['id'].tolist())
        for i in range(min_len):
            considered_seller = sellers_sorted.iloc[i, :]
            considered_buyer = buyers_sorted.iloc[i, :]

            if considered_buyer['last_offer'] >= considered_seller['last_offer']:
                # if seller price is lower or equal to buyer price
                # matching is performed
                agents.loc[agents['id'] == considered_buyer['id'], 'done'] = True
                agents.loc[agents['id'] == considered_seller['id'], 'done'] = True
                agents.loc[agents['id'] == considered_buyer['id'], 'stop_time'] = env_time
                agents.loc[agents['id'] == considered_seller['id'], 'stop_time'] = env_time

                deal_price = random.uniform(considered_seller['last_offer'], considered_buyer['last_offer'])

                if self.reward_on_reference:
                    rewards[considered_buyer['id']] = considered_buyer['res_price'] - considered_buyer['last_offer']
                    rewards[considered_seller['id']] = considered_seller['last_offer'] - considered_seller['res_price']
                else:
                    rewards[considered_buyer['id']] = considered_buyer['res_price'] - deal_price
                    rewards[considered_seller['id']] = deal_price - considered_seller['res_price']

                matching = dict(Seller=considered_seller['id'], Buyer=considered_buyer['id'], time=env_time, deal_price=deal_price)
                deal_history.append(matching)

            else:
                # not possible that new matches can occur after this failure due to sorting.
                break
        return rewards
