import pandas as pd
import numpy as np
from agents import Buyer, Seller
from environments import MarketEnvironment
from info_settings import InformationSetting
from matchers import RandomMatcher
from decision_maker import use_brain, determine_size_of_coefs
import warnings
# pandas setting warnings can be ignored, as it is intendend often
warnings.simplefilter("ignore")


n_sellers = 10
res_prices = np.random.normal(100, 5, n_sellers)
names = ['Seller ' + str(i) for i in range(1, n_sellers + 1)]
sellers = [Seller(names[i], res_prices[i]) for i in range(n_sellers)]

n_buyers = 10
res_prices = np.random.normal(200, 5, n_buyers)
names = ['Buyer ' + str(i) for i in range(1, n_buyers + 1)]
buyers = [Buyer(names[i], res_prices[i]) for i in range(n_buyers)]

setting = {
    'self_last_offer': True,
    'same_side_last_offers': True,
    'other_side_last_offers': True,
    'completed_deals': True,
    'current_time': True,
    'max_time': True,
    'n_sellers': True,
    'n_buyers': True
}
market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_steps=30,
                               matcher=RandomMatcher(reward_on_reference=True), setting=setting)
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
    print('---------------------------------------------------------------------------------------------------------------')

    for agent in sellers:
        size_coefs = determine_size_of_coefs(setting=setting, agent=agent, environment=market_env)
        coefs = np.ones(size_coefs)
        new_offer = use_brain(observations[agent.agent_id], agent=agent, environment=market_env, coefs=coefs)
        current_step_offers[agent.agent_id] = new_offer
    for agent in buyers:
        size_coefs = determine_size_of_coefs(setting=setting, agent=agent, environment=market_env)
        coefs = np.ones(size_coefs)
        new_offer = use_brain(observations[agent.agent_id], agent=agent, environment=market_env, coefs=coefs)
        current_step_offers[agent.agent_id] = new_offer
    print(current_step_offers)
    done = True


