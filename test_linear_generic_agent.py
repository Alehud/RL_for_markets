import pandas as pd
import numpy as np
from doubleauction.environments import MarketEnvironment
from doubleauction.matchers import RandomMatcher
from doubleauction.agents.linear_generic_agent import LinearGenericBuyer, LinearGenericSeller
import matplotlib.pyplot as plt
import warnings
# pandas setting warnings can be ignored, as it is intendend often
warnings.simplefilter("ignore")


n_sellers = 100
n_buyers = 100
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
sellers = np.array([LinearGenericSeller(agent_id=names[i], reservation_price=res_prices[i], n_sellers=n_sellers, n_buyers=n_buyers, max_time=max_time, setting=setting) for i in range(n_sellers)])


res_prices = np.random.normal(200, 5, n_buyers)
names = ['Buyer ' + str(i) for i in range(1, n_buyers + 1)]
buyers = np.array([LinearGenericBuyer(agent_id=names[i], reservation_price=res_prices[i], n_sellers=n_sellers, n_buyers=n_buyers, max_time=max_time, setting=setting) for i in range(n_buyers)])

market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_steps=max_time, matcher=RandomMatcher(reward_on_reference=True), setting=setting)
init_observation = market_env.reset()

# Initial offers
current_step_offers = {}
for agent in sellers:
    current_step_offers[agent.agent_id] = np.random.normal(200, 5)
for agent in buyers:
    current_step_offers[agent.agent_id] = np.random.normal(100, 5)
print(current_step_offers)

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 8), tight_layout=True)
ax.set_xlim(95, 205)
done = {"Dummy": False}
i = 1
while False in done.values():
    observations, rewards, done, _ = market_env.step(current_step_offers)
    done_sellers = np.array(list(done.values())[:n_sellers]) == False
    done_buyers = np.array(list(done.values())[n_sellers:]) == False
    print(i, '---------------------------------------------------------------------------------------------------------------')
    i += 1

    current_step_offers.clear()
    for agent in sellers[done_sellers]:
        size_coefs = agent.determine_size_of_coefs()
        coefs = np.array([0.05, 0.95, 0])
        new_offer = agent.decide(observations=observations[agent.agent_id], coefs=coefs)
        current_step_offers[agent.agent_id] = new_offer
    for agent in buyers[done_buyers]:
        size_coefs = agent.determine_size_of_coefs()
        coefs = np.array([0.05, 0.95, 0])
        new_offer = agent.decide(observations=observations[agent.agent_id], coefs=coefs)
        current_step_offers[agent.agent_id] = new_offer

    _, _, bars0 = ax.hist(list(current_step_offers.values()), 50, color='blue')
    plt.draw()
    plt.pause(0.1)
    _ = [b.remove() for b in bars0]

    print(current_step_offers)