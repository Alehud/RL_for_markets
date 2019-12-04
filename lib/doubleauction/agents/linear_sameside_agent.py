from doubleauction.agents import MarketAgent
import numpy as np
from abc import abstractmethod


class LinearSameSideAgent(MarketAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        Linear generic agent
        """
        super().__init__(agent_id, reservation_price)
        self.observations['previous_success'] = False

    def receive_observations_from_environment(self, env):
        agents = env.agents
        rewards = env.rewards

        self.reward += rewards[self.agent_id]

        agent_info = agents[agents['id'] == self.agent_id]
        agent_role = agent_info['role'].iloc[0]
        same_side_agents = agents[agents['role'] == agent_role]
        self_last_offer = agent_info['last_offer'].iloc[0]
        same_side_last_offers = np.array(same_side_agents.loc[same_side_agents['done'] == False]['last_offer'])
        same_side_not_done = len(same_side_agents.loc[same_side_agents['done'] == False])
        previous_success = agent_info['previous_success'].iloc[0]

        self.observations['self_last_offer'] = self_last_offer
        self.observations['same_side_last_offers'] = same_side_last_offers
        self.observations['same_side_not_done'] = same_side_not_done
        self.observations['current_time'] = env.time
        self.observations['previous_success'] = previous_success

    @abstractmethod
    def decide(self, n_sellers=None, n_buyers=None, max_time=None):
        return -1


class LinearSameSideBuyer(LinearSameSideAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        A buyer who takes determines the new offer as a linear combination of all data available in observation
        """
        super().__init__(agent_id, reservation_price)

    def compose_observation_vector(self, n_buyers: int):
        vals = np.array([self.reservation_price])
        vals = np.append(vals, self.observations['self_last_offer'])
        same_side_ofs = np.sort(self.observations['same_side_last_offers'])[::-1]
        same_side_ofs = np.concatenate((same_side_ofs, np.zeros(n_buyers - np.size(same_side_ofs))))
        vals = np.append(vals, same_side_ofs)
        vals = np.append(vals, self.observations['same_side_not_done'])
        vals = np.append(vals, self.observations['current_time'])
        vals = np.append(vals, self.observations['previous_success'])
        return vals

    def decide(self, n_sellers=None, n_buyers=None, max_time=None):
        vals = self.compose_observation_vector(n_buyers=n_buyers)
        new_offer = np.dot(vals, self.coefs)
        if new_offer > self.reservation_price:
            return self.reservation_price
        else:
            return new_offer

    def determine_size_of_coefs(self, n_buyers: int):
        # Reservation price is always known to agent
        size = 5 + n_buyers
        return int(size)


class LinearSameSideSeller(LinearSameSideAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        A seller who takes determines the new offer as a linear combination of all data available in observation
        """
        super().__init__(agent_id, reservation_price)

    def compose_observation_vector(self, n_sellers: int):
        vals = np.array([self.reservation_price])
        vals = np.append(vals, self.observations['self_last_offer'])
        same_side_ofs = np.sort(self.observations['same_side_last_offers'])[::-1]
        same_side_ofs = np.concatenate((same_side_ofs, np.zeros(n_sellers - np.size(same_side_ofs))))
        vals = np.append(vals, same_side_ofs)
        vals = np.append(vals, self.observations['same_side_not_done'])
        vals = np.append(vals, self.observations['current_time'])
        vals = np.append(vals, self.observations['previous_success'])
        return vals

    def decide(self, n_sellers=None, n_buyers=None, max_time=None):
        vals = self.compose_observation_vector(n_sellers=n_sellers)
        new_offer = np.dot(vals, self.coefs)
        if new_offer < self.reservation_price:
            return self.reservation_price
        else:
            return new_offer

    def determine_size_of_coefs(self, n_sellers: int):
        # Reservation price is always known to agent
        size = 5 + n_sellers
        return int(size)
