from doubleauction.agents import MarketAgent
import numpy as np
from abc import abstractmethod


class LinearBlackBoxAgent(MarketAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        Linear blackbox agent. Only self_last_offer, current_time and previous_success are known
        """
        super().__init__(agent_id, reservation_price)
        self.observations['previous_success'] = False

    def receive_observations_from_environment(self, env):
        agents = env.agents
        rewards = env.rewards

        self.reward += rewards[self.agent_id]

        agent_info = agents[agents['id'] == self.agent_id]
        self_last_offer = agent_info['last_offer'].iloc[0]
        previous_success = agent_info['previous_success'].iloc[0]

        self.observations['self_last_offer'] = self_last_offer
        self.observations['current_time'] = env.time
        self.observations['previous_success'] = previous_success

    @abstractmethod
    def decide(self, *args, **kwargs):
        return -1


class LinearBlackBoxBuyer(LinearBlackBoxAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        A buyer who takes determines the new offer as a linear combination of all data available in observation
        """
        super().__init__(agent_id, reservation_price)

    def compose_observation_vector(self):
        vals = np.array([self.reservation_price])
        vals = np.append(vals, self.observations['self_last_offer'])
        vals = np.append(vals, self.observations['current_time'])
        vals = np.append(vals, self.observations['previous_success'])
        return vals

    def decide(self):
        vals = self.compose_observation_vector()
        return np.dot(vals, self.coefs)


class LinearBlackBoxSeller(LinearBlackBoxAgent):
    def __init__(self, agent_id: str, reservation_price: float):
        """
        A seller who takes determines the new offer as a linear combination of all data available in observation
        """
        super().__init__(agent_id, reservation_price)

    def compose_observation_vector(self):
        vals = np.array([self.reservation_price])
        vals = np.append(vals, self.observations['self_last_offer'])
        vals = np.append(vals, self.observations['current_time'])
        vals = np.append(vals, self.observations['previous_success'])
        return vals

    def decide(self):
        vals = self.compose_observation_vector()
        return np.dot(vals, self.coefs)
