from doubleauction.agents import Buyer, Seller
import numpy as np


class NeuralNetworkBuyer(Buyer):
    def __init__(self, agent_id: str, reservation_price: float, hidden_layers: int, nodes_in_hidden_layers: list,
                 n_sellers: int, n_buyers: int, max_time: int, setting: dict):
        """
        A buyer who takes determines the new offer as a linear combination of all data available in observation
        """
        super().__init__(agent_id, reservation_price)
        self.n_sellers = n_sellers
        self.n_buyers = n_buyers
        self.max_time = max_time
        self.setting = setting
        self.hidden_layers = hidden_layers
        self.nodes_in_hidden_layers = nodes_in_hidden_layers
        if len(nodes_in_hidden_layers) != hidden_layers:
            raise Exception('nodes_in_hidden_layers should be a list of the size hidden_layers')

    def compose_observations(self, observations):
        max_amount_of_deals = min(self.n_sellers, self.n_buyers)
        vals = np.array([self.reservation_price])
        if 'self_last_offer' in observations:
            vals = np.append(vals, observations['self_last_offer'])
        if 'same_side_last_offers' in observations:
            same_side_ofs = np.sort(observations['same_side_last_offers'])[::-1]
            same_side_ofs = np.concatenate((same_side_ofs, np.zeros(self.n_buyers - np.size(same_side_ofs))))
            vals = np.append(vals, same_side_ofs)
        if 'other_side_last_offers' in observations:
            other_side_ofs = np.sort(observations['other_side_last_offers'])
            other_side_ofs = np.concatenate((other_side_ofs, np.zeros(self.n_sellers - np.size(other_side_ofs))))
            vals = np.append(vals, other_side_ofs)
        if 'completed_deals' in observations:
            sorted_by_time = np.array(sorted(observations['completed_deals'], key=lambda tup: tup[0])[::-1])
            if sorted_by_time.size == 0:
                vals = np.concatenate((vals, np.zeros(max_amount_of_deals * 2)))
            else:
                sorted_by_time[:, 0] = self.max_time - sorted_by_time[:, 0]
                sorted_by_time = sorted_by_time.flatten()
                vals = np.concatenate((vals, sorted_by_time))
                vals = np.concatenate((vals, np.zeros(max_amount_of_deals * 2 - np.size(sorted_by_time))))
        if 'current_time' in observations:
            vals = np.append(vals, observations['current_time'])
        if 'max_time' in observations:
            vals = np.append(vals, observations['max_time'])
        if 'n_sellers' in observations:
            vals = np.append(vals, observations['n_sellers'])
        if 'n_buyers' in observations:
            vals = np.append(vals, observations['n_buyers'])
        return vals

    def decide(self, observations, coefs):
        vals = self.compose_observations(observations)


        return np.dot(vals, coefs)

    def determine_size_of_coefs(self):
        size = 1
        if self.setting['self_last_offer']:
            size += 1
        if self.setting['same_side_last_offers']:
            size += self.n_buyers
        if self.setting['other_side_last_offers']:
            size += self.n_sellers
        if self.setting['completed_deals']:
            size += min(self.n_sellers, self.n_buyers) * 2
        if self.setting['current_time']:
            size += 1
        if self.setting['max_time']:
            size += 1
        if self.setting['n_sellers']:
            size += 1
        if self.setting['n_buyers']:
            size += 1
        return int(size)


class NeuralNetworkSeller(Seller):
    def __init__(self, agent_id: str, reservation_price: float, hidden_layers: int, nodes_in_hidden_layers: list,
                 n_sellers: int, n_buyers: int, max_time: int, setting: dict):
        """
        A seller who takes determines the new offer as a linear combination of all data available in observation
        """
        super().__init__(agent_id, reservation_price)
        self.n_sellers = n_sellers
        self.n_buyers = n_buyers
        self.max_time = max_time
        self.setting = setting
        self.hidden_layers = hidden_layers
        self.nodes_in_hidden_layers = nodes_in_hidden_layers
        if len(nodes_in_hidden_layers) != hidden_layers:
            raise Exception('nodes_in_hidden_layers should be a list of the size hidden_layers')

    def compose_observations(self, observations):
        max_amount_of_deals = min(self.n_sellers, self.n_buyers)
        vals = np.array([self.reservation_price])
        if 'self_last_offer' in observations:
            vals = np.append(vals, observations['self_last_offer'])
        if 'same_side_last_offers' in observations:
            same_side_ofs = np.sort(observations['same_side_last_offers'])
            same_side_ofs = np.concatenate((same_side_ofs, np.zeros(self.n_sellers - np.size(same_side_ofs))))
            vals = np.append(vals, same_side_ofs)
        if 'other_side_last_offers' in observations:
            other_side_ofs = np.sort(observations['other_side_last_offers'])[::-1]
            other_side_ofs = np.concatenate((other_side_ofs, np.zeros(self.n_buyers - np.size(other_side_ofs))))
            vals = np.append(vals, other_side_ofs)
        if 'completed_deals' in observations:
            sorted_by_time = np.array(sorted(observations['completed_deals'], key=lambda tup: tup[0])[::-1])
            if sorted_by_time.size == 0:
                vals = np.concatenate((vals, np.zeros(max_amount_of_deals * 2)))
            else:
                sorted_by_time[:, 0] = self.max_time - sorted_by_time[:, 0]
                sorted_by_time = sorted_by_time.flatten()
                vals = np.concatenate((vals, sorted_by_time))
                vals = np.concatenate((vals, np.zeros(max_amount_of_deals * 2 - np.size(sorted_by_time))))
        if 'current_time' in observations:
            vals = np.append(vals, observations['current_time'])
        if 'max_time' in observations:
            vals = np.append(vals, observations['max_time'])
        if 'n_sellers' in observations:
            vals = np.append(vals, observations['n_sellers'])
        if 'n_buyers' in observations:
            vals = np.append(vals, observations['n_buyers'])
        return vals

    def decide(self, observations, coefs):
        vals = self.compose_observations(observations)
        return np.dot(vals, coefs)

    def determine_size_of_coefs(self):
        size = 1
        if self.setting['self_last_offer']:
            size += 1
        if self.setting['same_side_last_offers']:
            size += self.n_sellers
        if self.setting['other_side_last_offers']:
            size += self.n_buyers
        if self.setting['completed_deals']:
            size += min(self.n_sellers, self.n_buyers) * 2
        if self.setting['current_time']:
            size += 1
        if self.setting['max_time']:
            size += 1
        if self.setting['n_sellers']:
            size += 1
        if self.setting['n_buyers']:
            size += 1
        return int(size)
