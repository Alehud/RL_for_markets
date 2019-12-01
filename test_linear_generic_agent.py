import pandas as pd
import numpy as np
from doubleauction.environments import MarketEnvironment
from doubleauction.matchers import RandomMatcher
from doubleauction.agents.linear_generic_agent import LinearGenericBuyer, LinearGenericSeller
import matplotlib.pyplot as plt
import warnings
# pandas setting warnings can be ignored, as it is intendend often
warnings.simplefilter("ignore")


# Define the initial number of agents
n_sellers = 100
n_buyers = 100

# Create initial agents
# All agents are the same for now: linear_generic_agents with such information setting:
setting = {
    'self_last_offer': True,
    'same_side_last_offers': False,
    'other_side_last_offers': False,
    'completed_deals': False,
    'current_time': True,
    'max_time': False,
    'n_sellers': False,
    'n_buyers': False,
    'previous_round_success': True
}
res_prices = np.random.normal(100, 5, n_sellers)
names = ['Seller ' + str(i) for i in range(1, n_sellers + 1)]
sellers = np.array([LinearGenericSeller(agent_id=names[i], reservation_price=res_prices[i], setting=setting) for i in range(n_sellers)])
res_prices = np.random.normal(200, 5, n_buyers)
names = ['Buyer ' + str(i) for i in range(1, n_buyers + 1)]
buyers = np.array([LinearGenericBuyer(agent_id=names[i], reservation_price=res_prices[i], setting=setting) for i in range(n_buyers)])

# We're playing only 1 game for now, otherwise here would a loop over games

# Define parameters of a round
max_time = 30
matcher = RandomMatcher(reward_on_reference=True)

# Create market environment
market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_steps=max_time, matcher=matcher, setting=setting)

# Define/change coefs for agents
for agent in sellers:
    size_coefs = agent.determine_size_of_coefs(n_buyers=n_buyers, n_sellers=n_sellers)
    # Coefs are generated here somehow
    agent.coefs = np.array([0.05, 0.95, 0, 0])
for agent in buyers:
    size_coefs = agent.determine_size_of_coefs(n_buyers=n_buyers, n_sellers=n_sellers)
    # Coefs are generated here somehow
    agent.coefs = np.array([0.05, 0.95, 0, 0])

# Loop over rounds
for round in range(10):
    print("ROUND", round)

    init_observation = market_env.reset()

    # Initial offers are generated
    current_step_offers = {}
    for agent in sellers:
        current_step_offers[agent.agent_id] = np.random.normal(200, 5)
    for agent in buyers:
        current_step_offers[agent.agent_id] = np.random.normal(100, 5)
    print(current_step_offers)

    # for plotting
    # fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 8), tight_layout=True)
    # ax.set_xlim(95, 205)

    done = {"Dummy": False}
    i = 1

    # Loop over time steps
    while (False in done.values()) and (i <= max_time):
        # Environment calculates what happens and generates some observations
        observations, rewards, done, _ = market_env.step(current_step_offers)
        not_done_sellers = np.array(list(done.values())[:n_sellers]) == False
        not_done_buyers = np.array(list(done.values())[n_sellers:]) == False
        print('Time step', i, '---------------------------------------------------------------------------------------------------------------')
        i += 1

        # Clearing current offers
        current_step_offers.clear()

        # All agents receive observations from what environment generated
        for agent in sellers:
            agent.receive_observations_from_environment(observations[agent.agent_id])
        for agent in buyers:
            agent.receive_observations_from_environment(observations[agent.agent_id])

        # Agents who are not done yet decide on a new offer
        for agent in sellers[not_done_sellers]:
            new_offer = agent.decide(n_sellers=n_sellers, n_buyers=n_buyers, max_time=max_time)
            # His offer is put in the list of offers
            current_step_offers[agent.agent_id] = new_offer
        for agent in buyers[not_done_buyers]:
            new_offer = agent.decide(n_sellers=n_sellers, n_buyers=n_buyers, max_time=max_time)
            current_step_offers[agent.agent_id] = new_offer

        # for plotting
        # _, _, bars0 = ax.hist(list(current_step_offers.values()), 50, color='blue')
        # plt.draw()
        # plt.pause(0.1)
        # _ = [b.remove() for b in bars0]

        print("Agents still in play:", current_step_offers)

