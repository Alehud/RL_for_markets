from doubleauction.agents import Buyer, Seller, MarketAgent
import numpy as np
from abc import abstractmethod


class LinearGenericAgent(MarketAgent):
    def __init__(self, agent_id: str, reservation_price: float, setting: dict):
        """
        Linear generic agent
        """
        super().__init__(agent_id, reservation_price)
        self.setting = setting
        if self.setting['previous_success']:
            self.observations['previous_success'] = False

    def receive_observations_from_environment(self, observations):
        if self.setting['self_last_offer']:
            self.observations['self_last_offer'] = observations['self_last_offer']
        if self.setting['same_side_last_offers']:
            self.observations['same_side_last_offers'] = observations['same_side_last_offers']
        if self.setting['same_side_res_prices']:
            self.observations['same_side_res_prices'] = observations['same_side_res_prices']
        if self.setting['same_side_not_done']:
            self.observations['same_side_not_done'] = observations['same_side_not_done']
        if self.setting['other_side_last_offers']:
            self.observations['other_side_last_offers'] = observations['other_side_last_offers']
        if self.setting['other_side_res_prices']:
            self.observations['other_side_res_prices'] = observations['other_side_res_prices']
        if self.setting['other_side_not_done']:
            self.observations['other_side_not_done'] = observations['other_side_not_done']
        if self.setting['completed_deals']:
            self.observations['completed_deals'] = observations['completed_deals']
        if self.setting['current_time']:
            self.observations['current_time'] = observations['current_time']
        if self.setting['max_time']:
            self.observations['max_time'] = observations['max_time']
        if self.setting['n_buyers']:
            self.observations['n_buyers'] = observations['n_buyers']
        if self.setting['n_sellers']:
            self.observations['n_sellers'] = observations['n_sellers']
        if self.setting['previous_success']:
            self.observations['previous_success'] = observations['previous_success']

    @abstractmethod
    def decide(self, n_sellers=None, n_buyers=None, max_time=None):
        return -1


class LinearGenericBuyer(LinearGenericAgent):
    def __init__(self, agent_id: str, reservation_price: float, setting: dict):
        """
        A buyer who takes determines the new offer as a linear combination of all data available in observation
        """
        super().__init__(agent_id, reservation_price, setting)

    def compose_observation_vector(self, n_sellers: int, n_buyers: int, max_time: int):
        max_amount_of_deals = min(n_sellers, n_buyers)
        vals = np.array([self.reservation_price])
        if self.setting['self_last_offer']:
            vals = np.append(vals, self.observations['self_last_offer'])
        if self.setting['same_side_last_offers']:
            same_side_ofs = np.sort(self.observations['same_side_last_offers'])[::-1]
            same_side_ofs = np.concatenate((same_side_ofs, np.zeros(n_buyers - np.size(same_side_ofs))))
            vals = np.append(vals, same_side_ofs)
        if self.setting['other_side_last_offers']:
            other_side_ofs = np.sort(self.observations['other_side_last_offers'])
            other_side_ofs = np.concatenate((other_side_ofs, np.zeros(n_sellers - np.size(other_side_ofs))))
            vals = np.append(vals, other_side_ofs)
        if self.setting['completed_deals']:
            sorted_by_time = np.array(sorted(self.observations['completed_deals'], key=lambda tup: tup[0])[::-1])
            if sorted_by_time.size == 0:
                vals = np.concatenate((vals, np.zeros(max_amount_of_deals * 2)))
            else:
                sorted_by_time[:, 0] = max_time - sorted_by_time[:, 0]
                sorted_by_time = sorted_by_time.flatten()
                vals = np.concatenate((vals, sorted_by_time))
                vals = np.concatenate((vals, np.zeros(max_amount_of_deals * 2 - np.size(sorted_by_time))))
        if self.setting['current_time']:
            vals = np.append(vals, self.observations['current_time'])
        if self.setting['max_time']:
            vals = np.append(vals, self.observations['max_time'])
        if self.setting['n_sellers']:
            vals = np.append(vals, self.observations['n_sellers'])
        if self.setting['n_buyers']:
            vals = np.append(vals, self.observations['n_buyers'])
        if self.setting['previous_success']:
            vals = np.append(vals, self.observations['previous_success'])
        return vals

    def decide(self, n_sellers=None, n_buyers=None, max_time=None):
        vals = self.compose_observation_vector(n_sellers=n_sellers, n_buyers=n_buyers, max_time=max_time)
        return np.dot(vals, self.coefs)

    def determine_size_of_coefs(self, n_sellers: int, n_buyers: int):
        # Reservation price is always known to agent
        size = 1
        if self.setting['self_last_offer']:
            size += 1
        if self.setting['same_side_last_offers']:
            size += n_buyers
        if self.setting['other_side_last_offers']:
            size += n_sellers
        if self.setting['completed_deals']:
            size += min(n_sellers, n_buyers) * 2
        if self.setting['current_time']:
            size += 1
        if self.setting['max_time']:
            size += 1
        if self.setting['n_sellers']:
            size += 1
        if self.setting['n_buyers']:
            size += 1
        if self.setting['previous_success']:
            size += 1
        return int(size)


class LinearGenericSeller(LinearGenericAgent):
    def __init__(self, agent_id: str, reservation_price: float, setting: dict):
        """
        A seller who takes determines the new offer as a linear combination of all data available in observation
        """
        super().__init__(agent_id, reservation_price, setting)

    def compose_observation_vector(self, n_sellers: int, n_buyers: int, max_time: int):
        max_amount_of_deals = min(n_sellers, n_buyers)
        vals = np.array([self.reservation_price])
        if self.setting['self_last_offer']:
            vals = np.append(vals, self.observations['self_last_offer'])
        if self.setting['same_side_last_offers']:
            same_side_ofs = np.sort(self.observations['same_side_last_offers'])[::-1]
            same_side_ofs = np.concatenate((same_side_ofs, np.zeros(n_sellers - np.size(same_side_ofs))))
            vals = np.append(vals, same_side_ofs)
        if self.setting['other_side_last_offers']:
            other_side_ofs = np.sort(self.observations['other_side_last_offers'])
            other_side_ofs = np.concatenate((other_side_ofs, np.zeros(n_buyers - np.size(other_side_ofs))))
            vals = np.append(vals, other_side_ofs)
        if self.setting['completed_deals']:
            sorted_by_time = np.array(sorted(self.observations['completed_deals'], key=lambda tup: tup[0])[::-1])
            if sorted_by_time.size == 0:
                vals = np.concatenate((vals, np.zeros(max_amount_of_deals * 2)))
            else:
                sorted_by_time[:, 0] = max_time - sorted_by_time[:, 0]
                sorted_by_time = sorted_by_time.flatten()
                vals = np.concatenate((vals, sorted_by_time))
                vals = np.concatenate((vals, np.zeros(max_amount_of_deals * 2 - np.size(sorted_by_time))))
        if self.setting['current_time']:
            vals = np.append(vals, self.observations['current_time'])
        if self.setting['max_time']:
            vals = np.append(vals, self.observations['max_time'])
        if self.setting['n_sellers']:
            vals = np.append(vals, self.observations['n_sellers'])
        if self.setting['n_buyers']:
            vals = np.append(vals, self.observations['n_buyers'])
        if self.setting['previous_success']:
            vals = np.append(vals, self.observations['previous_success'])
        return vals

    def decide(self, n_sellers=None, n_buyers=None, max_time=None):
        vals = self.compose_observation_vector(n_sellers=n_sellers, n_buyers=n_buyers, max_time=max_time)
        return np.dot(vals, self.coefs)

    def determine_size_of_coefs(self, n_sellers: int, n_buyers: int):
        # Reservation price is always known to agent
        size = 1
        if self.setting['self_last_offer']:
            size += 1
        if self.setting['same_side_last_offers']:
            size += n_sellers
        if self.setting['other_side_last_offers']:
            size += n_buyers
        if self.setting['completed_deals']:
            size += min(n_sellers, n_buyers) * 2
        if self.setting['current_time']:
            size += 1
        if self.setting['max_time']:
            size += 1
        if self.setting['n_sellers']:
            size += 1
        if self.setting['n_buyers']:
            size += 1
        if self.setting['previous_success']:
            size += 1
        return int(size)
