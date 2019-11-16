__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"


from gym.spaces import Box
import numpy as np
import pandas as pd
from abc import abstractmethod


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
        pass

    @abstractmethod
    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame,
                  offers: pd.DataFrame):
        """
        The method that generates the state for the agents, based on the information setting.
        :param agent_id: usually a string, a unique id for an agent
        :param deal_history: the dictionary containing all the successful deals till now
        :param agents: the dataframe containing the agent information
        :param offers: The dataframe containing the past offers from agents
        :return: In the abstract case, a zero value scalar is retuned.
        """
        return np.zeros(1)


class BlackBoxSetting(InformationSetting):
    def __init__(self, agents):
        """
        The agent is only aware about their past actions, which some positive real value
        representing an offer.
        :param agents: The dataframe of agents in the environment.
        """
        super().__init__(agents)

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame, offers: pd.DataFrame):
        self_last_offer = np.array(offers.loc[offers['id'] == agent_id]['offer'])

        observation_state = dict(self_last_offer=self_last_offer)
        return observation_state


class SameSideSetting(InformationSetting):
    def __init__(self, agents):
        """
        The agent is only aware about the last offers submitted by agents sharing the same role.
        The observation for each agent is a vector number of agents dimensions, which contains
        positive
        values for agents of the same role, and zero for agents of the other side.
        :param agents: The dataframe of agents in the environment.
        """
        super().__init__(agents)

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame, offers: pd.DataFrame):
        self_last_offer = np.array(offers.loc[offers['id'] == agent_id]['offer'])
        agent_role = agents.loc[agents['id'] == agent_id]['role'].iloc[0]
        same_side_last_offers = np.array(offers.loc[offers['role'] == agent_role]['offer'])

        observation_state = dict(self_last_offer=self_last_offer, same_side_last_offers=same_side_last_offers)
        return observation_state


class OtherSideSetting(InformationSetting):
    def __init__(self, agents):
        """
        The agent is only aware about the last offers submitted by agents sharing the other role.
        The observation for each agent is a vector number of agents dimensions, which contains
        positive
        values for agents of the other role, and zero for agents of the same side.
        :param agents: The dataframe of agents in the environment.
        """
        super().__init__(agents)

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame, offers: pd.DataFrame):
        self_last_offer = np.array(offers.loc[offers['id'] == agent_id]['offer'])
        agent_role = agents.loc[agents['id'] == agent_id]['role'].iloc[0]
        other_side_last_offers = np.array(offers.loc[offers['role'] != agent_role]['offer'])

        observation_state = dict(self_last_offer=self_last_offer, other_side_last_offers=other_side_last_offers)
        return observation_state


class BothSidesSetting(InformationSetting):
    def __init__(self, agents):
        """
        The agent is aware about the all offers submitted by agents.
        The observation for each agent is a vector number of agents dimensions, which contains
        positive
        values for offers of all agents.
        :param agents: The dataframe of agents in the environment.
        """
        super().__init__(agents)

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame, offers: pd.DataFrame):
        self_last_offer = np.array(offers.loc[offers['id'] == agent_id]['offer'])
        agent_role = agents.loc[agents['id'] == agent_id]['role'].iloc[0]
        same_side_last_offers = np.array(offers.loc[offers['role'] == agent_role]['offer'])
        other_side_last_offers = np.array(offers.loc[offers['role'] != agent_role]['offer'])

        observation_state = dict(self_last_offer=self_last_offer, same_side_last_offers=same_side_last_offers,
                                 other_side_last_offers=other_side_last_offers)
        return observation_state


class DealInformationSetting(InformationSetting):
    def __init__(self, agents):
        """
        The agent is aware about the all deal values in the current round.
        The observation for each agent is a vector of dimensions equal to the minimum number
        of agents having the same role. The vector contains positive values for all successful
        offers and zeros otherwise.
        :param agents: The dataframe of agents in the environment.
        """
        super().__init__(agents)

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame, offers: pd.DataFrame):
        self_last_offer = np.array(offers.loc[offers['id'] == agent_id]['offer'])
        completed_deals = [x['deal_price'] for x in deal_history]

        observation_state = dict(self_last_offer=self_last_offer, completed_deals=completed_deals)
        return observation_state


class FullInformationSetting(InformationSetting):
    def __init__(self, agents):
        """
        The agent is aware about the all deal values in the current round and all agent offers.
        The observation for each agent is a vector of dimensions equal to the minimum number
        of agents having the same role plus the number of agents.
        The vector contains positive values for all successful offers and zeros otherwise.
        :param agents: The dataframe of agents in the environment.
        """
        super().__init__(agents)

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame, offers: pd.DataFrame):
        self_last_offer = np.array(offers.loc[offers['id'] == agent_id]['offer'])
        agent_role = agents.loc[agents['id'] == agent_id]['role'].iloc[0]
        same_side_last_offers = np.array(offers.loc[offers['role'] == agent_role]['offer'])
        other_side_last_offers = np.array(offers.loc[offers['role'] != agent_role]['offer'])
        completed_deals = [x['deal_price'] for x in deal_history]

        observation_state = dict(self_last_offer=self_last_offer, same_side_last_offers=same_side_last_offers,
                                 other_side_last_offers=other_side_last_offers, completed_deals=completed_deals)
        return observation_state
