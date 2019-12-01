import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import warnings
# pandas setting warnings can be ignored, as it is intendend often
warnings.simplefilter("ignore")
from doubleauction.agents import Buyer, Seller
from doubleauction.environments import MarketEnvironment
import doubleauction.info_settings as info
from doubleauction.matchers import RandomMatcher


def generate_offers(market_environment, obs, param):
    # make param unique for each agent

    # obs: initial observations
    # param could be a list if several parameters exist (each of them <= 1)
    # how to make sure it stays in the sample space? (not exceeding/going below reservation price)

    id_offer = []

    for id in market_environment.agent_ids:
        if id in market_environment.matched:
            continue

        role = market_environment.agent_roles.get(id)
        df = market_environment.agents
        res_price = df.loc[df['id'] == str(id), 'res_price'].iloc[0]
        offer_tmp = []

        self_last_offer = obs.get(id).get('self_last_offer')
        if role == 'Seller':
            offer_tmp1 = self_last_offer - (self_last_offer - res_price) * param[0]
        elif role == 'Buyer':
            offer_tmp1 = (res_price - self_last_offer) * param[0] + self_last_offer
        else:
            break
        offer_tmp.append(offer_tmp1)

        key = 'same_side_last_offers'
        if key in obs[str(id)].keys():
            same_side_offer = obs[str(id)].get(key)

            if role == 'Seller':
                target = np.sort(same_side_offer)[0]        # lowest seller
                offer_tmp2 = self_last_offer - (self_last_offer - target) * param[0]
            elif role == 'Buyer':
                target = np.sort(same_side_offer)[-1]       # highest buyer
                offer_tmp2 = self_last_offer + (target - self_last_offer) * param[0]
            else:
                break
            offer_tmp.append(offer_tmp2)

        key = 'other_side_last_offers'
        if key in obs[str(id)].keys():
            other_side_offer = obs[str(id)].get(key)

            if role == 'Seller':
                target = np.sort(other_side_offer)[-1]  # highest buyer
                offer_tmp3 = self_last_offer - (self_last_offer - target) * param[0]

            elif role == 'Buyer':
                target = np.sort(other_side_offer)[0]  # lowest seller
                offer_tmp3 = self_last_offer + (target - self_last_offer) * param[0]
            else:
                break
            offer_tmp.append(offer_tmp3)

        avg_new_offer = sum(offer_tmp) / len(offer_tmp)

        if role == 'Seller':
            theta = avg_new_offer - res_price
            while theta < 0:
                avg_new_offer += abs(theta) * (1 + param[1])
                theta = avg_new_offer - res_price
        elif role == 'Buyer':
            theta = res_price - avg_new_offer
            while theta < 0:
                avg_new_offer -= abs(theta) * (1 + param[1])
                theta = res_price - avg_new_offer
        else:
            break
        offer = avg_new_offer

        id_offer.append((id, offer))

    return dict(id_offer)


def complete_set(initial_offers, market_environment, param):
    observations, rewards, done, _ = market_environment.step(initial_offers)

    i = 0

    final_rewards = []

    while i < market_environment.max_steps:
        for id in market_environment.agent_ids:
            reward = rewards.get(id)
            if reward:
                final_rewards.append((id, dict((('time', i), ('reward', reward)))))

        # deal_hist = pd.DataFrame(market_environment.deal_history)
        # print("deal history: ")
        # print(deal_hist)
        new_off = generate_offers(market_environment, observations, param)
        observations, rewards, done, _ = market_env.step(new_off)

        # print("no. sets: ", i)
        # print("new offer: ", new_off)
        # print("rewards: ", rewards)
        i += 1

    return dict(final_rewards)


# playground starts here
john = Seller('Seller John', 100)
nick = Seller('Seller Nick', 110)
bob = Seller('Seller Bob', 140)
sellers = [john, nick, bob]

alex = Buyer('Buyer Alex', 130)
kevin = Buyer('Buyer Kevin', 110)
neri = Buyer('Buyer Neri', 150)
caterina = Buyer('Buyer Caterina', 100)
buyers = [alex, kevin, neri, caterina]

settings = {'self_last_offer': True,
            'same_side_last_offers': 1,
            'other_side_last_offers': True,
            'completed_deals': True,
            'current_time': True,
            'max_time': True,
            'n_sellers': True,
            'n_buyers': True}

market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_steps=20,
                               matcher=RandomMatcher(reward_on_reference=1), setting=settings)
# rewards on reference set to true
init_observation = market_env.reset()

init_offers = {
    'Buyer Alex': alex.reservation_price - 10.0,
    'Buyer Kevin': kevin.reservation_price - 5.0,
    'Seller John': john.reservation_price + 10.0,
    'Seller Nick': nick.reservation_price + 15.0,
    'Seller Bob': bob.reservation_price + 20.,
    'Buyer Neri': neri.reservation_price - 3.}

param_list = [0.6, 0.5]
final = complete_set(init_offers, market_env, param_list)
print(final)
