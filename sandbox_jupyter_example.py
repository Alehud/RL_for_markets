import pandas as pd
import numpy as np
from environments import MarketEnvironment
from matchers import RandomMatcher
from linear_generic_agent import LinearGenericBuyer, LinearGenericSeller
import warnings
# pandas setting warnings can be ignored, as it is intendend often
warnings.simplefilter("ignore")


n_sellers = 10
n_buyers = 10
max_time = 30
setting = {
    'self_last_offer': True,
    'same_side_last_offers': False,
    'other_side_last_offers': False,
    'completed_deals': False,
    'current_time': True,
    'max_time': False,
    'n_sellers': False,
    'n_buyers': False
}


res_prices = np.random.normal(100, 5, n_sellers)
names = ['Seller ' + str(i) for i in range(1, n_sellers + 1)]
sellers = [LinearGenericSeller(agent_id=names[i], reservation_price=res_prices[i], n_sellers=n_sellers, n_buyers=n_buyers, max_time=max_time, setting=setting) for i in range(n_sellers)]


res_prices = np.random.normal(200, 5, n_buyers)
names = ['Buyer ' + str(i) for i in range(1, n_buyers + 1)]
buyers = [LinearGenericBuyer(agent_id=names[i], reservation_price=res_prices[i], n_sellers=n_sellers, n_buyers=n_buyers, max_time=max_time, setting=setting) for i in range(n_buyers)]

market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_steps=max_time, matcher=RandomMatcher(reward_on_reference=True), setting=setting)
init_observation = market_env.reset()

# Initial offers
current_step_offers = {}
for agent in sellers:
    current_step_offers[agent.agent_id] = np.random.normal(200, 5)
for agent in buyers:
    current_step_offers[agent.agent_id] = np.random.normal(100, 5)

done = False
while not done:
    observations, rewards, done, _ = market_env.step(current_step_offers)
    print(observations)
    print('---------------------------------------------------------------------------------------------------------------')

    for agent in sellers:
        size_coefs = agent.determine_size_of_coefs()
        coefs = np.ones(size_coefs)
        new_offer = agent.decide(observations=observations[agent.agent_id], coefs=coefs)
        current_step_offers[agent.agent_id] = new_offer
    for agent in buyers:
        size_coefs = agent.determine_size_of_coefs()
        coefs = np.ones(size_coefs)
        new_offer = agent.decide(observations=observations[agent.agent_id], coefs=coefs)
        current_step_offers[agent.agent_id] = new_offer
    print(current_step_offers)
    done = True


