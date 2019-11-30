__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"


import numpy as np
import pandas as pd


class InformationSetting:
    def __init__(self, agents):
        """
        An abstract implementation of an information setting.
        Usually an observation space given the gym specification is provided, to enable
        describing of how a single agent observation looks like.
        Most of the time it will be a box environment, constraint between
        [low, high], with dimension defined by parameter shape.
        :param agents:
        """
        self.agents = agents

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame,
                  offers: pd.DataFrame, matched: set, current_time, max_time, n_sellers, n_buyers, bool_dict: dict):
        self_last_offer = np.array(offers.loc[offers['id'] == agent_id]['offer'])[0]
        agent_role = agents.loc[agents['id'] == agent_id]['role'].iloc[0]
        not_matched_offers = offers.loc[offers['id'].isin(matched) == False]
        same_side_last_offers = np.array(not_matched_offers.loc[not_matched_offers['role'] == agent_role]['offer'])
        other_side_last_offers = np.array(not_matched_offers.loc[not_matched_offers['role'] != agent_role]['offer'])
        completed_deals = [(x['time'], x['deal_price']) for x in deal_history]

        observation_state = dict()
        if bool_dict['self_last_offer']:
            observation_state['self_last_offer'] = self_last_offer
        if bool_dict['same_side_last_offers']:
            observation_state['same_side_last_offers'] = same_side_last_offers
        if bool_dict['other_side_last_offers']:
            observation_state['other_side_last_offers'] = other_side_last_offers
        if bool_dict['completed_deals']:
            observation_state['completed_deals'] = completed_deals
        if bool_dict['current_time']:
            observation_state['current_time'] = current_time
        if bool_dict['max_time']:
            observation_state['max_time'] = max_time
        if bool_dict['n_sellers']:
            observation_state['n_sellers'] = n_sellers
        if bool_dict['n_buyers']:
            observation_state['n_buyers'] = n_buyers
        return observation_state

        """
        The method that generates the state for the agents, based on the information setting.
        :param agent_id: usually a string, a unique id for an agent
        :param deal_history: the dictionary containing all the successful deals till now
        :param agents: the dataframe containing the agent information
        :param offers: The dataframe containing the past offers from agents
        :param current_time: integer, current time in the game
        :param max_time: integer, maximum allowed time of the game
        :param n_sellers: integer, the amount of sellers
        :param n_buyers: integer, the amount of buyers
        :param bool_dict: a dictionary with boolean values containing information about which observations should be turned on/off
        :return: the observation space
        """
