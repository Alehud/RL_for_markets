import pandas as pd
import numpy as np
from environments import MarketEnvironment
from info_settings import InformationSetting
from matchers import RandomMatcher


def use_brain(observation, agent, environment, coefs):
    agent_role = environment.agents.loc[environment.agents['id'] == agent.agent_id]['role'].iloc[0]
    num_sellers = environment.n_sellers
    num_buyers = environment.n_buyers
    max_amount_of_deals = min(num_sellers, num_buyers)
    vals = np.array([agent.reservation_price])

    if 'self_last_offer' in observation:
        vals = np.append(vals, observation['self_last_offer'])
    if 'same_side_last_offers' in observation:
        if agent_role == 'Seller':
            same_side_ofs = np.sort(observation['same_side_last_offers'])
            same_side_ofs = np.concatenate((same_side_ofs, np.zeros(num_sellers - np.size(same_side_ofs))))
            vals = np.append(vals, same_side_ofs)
        if agent_role == 'Buyer':
            same_side_ofs = np.sort(observation['same_side_last_offers'])[::-1]
            same_side_ofs = np.concatenate((same_side_ofs, np.zeros(num_buyers - np.size(same_side_ofs))))
            vals = np.append(vals, same_side_ofs)
    if 'other_side_last_offers' in observation:
        if agent_role == 'Seller':
            other_side_ofs = np.sort(observation['other_side_last_offers'])[::-1]
            other_side_ofs = np.concatenate((other_side_ofs, np.zeros(num_buyers - np.size(other_side_ofs))))
            vals = np.append(vals, other_side_ofs)
        if agent_role == 'Buyer':
            other_side_ofs = np.sort(observation['other_side_last_offers'])
            other_side_ofs = np.concatenate((other_side_ofs, np.zeros(num_sellers - np.size(other_side_ofs))))
            vals = np.append(vals, other_side_ofs)
    if 'completed_deals' in observation:
        sorted_by_time = np.array(sorted(observation['completed_deals'], key=lambda tup: tup[0])[::-1])
        if sorted_by_time.size == 0:
            vals = np.concatenate((vals, np.zeros(max_amount_of_deals * 2)))
        else:
            sorted_by_time[:, 0] = environment.time - sorted_by_time[:, 0]
            sorted_by_time = sorted_by_time.flatten()
            vals = np.concatenate((vals, sorted_by_time))
            vals = np.concatenate((vals, np.zeros(max_amount_of_deals*2 - np.size(sorted_by_time))))

    if 'current_time' in observation:
        vals = np.append(vals, observation['current_time'])
    if 'max_time' in observation:
        vals = np.append(vals, observation['max_time'])
    if 'n_sellers' in observation:
        vals = np.append(vals, observation['n_sellers'])
    if 'n_buyers' in observation:
        vals = np.append(vals, observation['n_buyers'])

    return np.dot(vals, coefs)


def determine_size_of_coefs(setting: dict, agent, environment):
    size = 1
    agent_role = environment.agents.loc[environment.agents['id'] == agent.agent_id]['role'].iloc[0]

    if setting['self_last_offer']:
        size += 1
    if setting['same_side_last_offers']:
        if agent_role == 'Seller':
            size += environment.n_sellers
        if agent_role == 'Buyer':
            size += environment.n_buyers
    if setting['other_side_last_offers']:
        if agent_role == 'Seller':
            size += environment.n_buyers
        if agent_role == 'Buyer':
            size += environment.n_sellers
    if setting['completed_deals']:
        size += min(environment.n_sellers, environment.n_buyers) * 2
    if setting['current_time']:
        size += 1
    if setting['max_time']:
        size += 1
    if setting['n_sellers']:
        size += 1
    if setting['n_buyers']:
        size += 1
    return int(size)
