import numpy as np
from doubleauction.environments import MarketEnvironment
from doubleauction.matchers import RandomMatcher
from doubleauction.util import generate_buyer_prices_paper, generate_seller_prices_paper
from doubleauction.agents.linear_blackbox_agent import LinearBlackBoxBuyer, LinearBlackBoxSeller
import matplotlib.pyplot as plt
import time
import warnings
# pandas setting warnings can be ignored, as it is intendend often
warnings.simplefilter("ignore")

start = time.time()

# Define the initial number of agents, the number of rounds and games
n_sellers = 100
n_buyers = 100
n_game = 1
n_round = 10


# Create initial agents with names and reservation prices
# All agents are the same for now
res_prices = generate_seller_prices_paper(discrete=False, count=n_sellers)
names = ['Seller ' + str(i) for i in range(1, n_sellers + 1)]
sellers = np.array([LinearBlackBoxSeller(agent_id=names[i], reservation_price=res_prices[i], noisy=True) for i in range(n_sellers)])
res_prices = generate_buyer_prices_paper(discrete=False, count=n_buyers)
names = ['Buyer ' + str(i) for i in range(1, n_buyers + 1)]
buyers = np.array([LinearBlackBoxBuyer(agent_id=names[i], reservation_price=res_prices[i], noisy=True) for i in range(n_buyers)])


# For plotting
# fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
# ax.set_xlim(75, 150)

# Loop over games
for g in range(1):
    print("GAME", g, '=================================================================================================================')

    # Define parameters of each round
    max_time = 50
    matcher = RandomMatcher(reward_on_reference=False)

    # Create market environment
    market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_time=max_time, matcher=matcher)

    # HERE AGENTS LEARN AND ADJUST THEIR COEFS (for now the are constant)
    for agent in sellers:
        agent.coefs = np.array([1-0.75497335, 0.75497335, 2.5])
    for agent in buyers:
        agent.coefs = np.array([1-0.75497335, 0.75497335, -2.5])

    # Reset agents' rewards and observations
    for agent in sellers:
        agent.reward = 0.0
        agent.observations = {}
    for agent in buyers:
        agent.reward = 0.0
        agent.observations = {}

    # Loop over rounds
    for r in range(3):
        print("ROUND", r, '-----------------------------------------------')

        # Reset market environment
        market_env.reset()

        # Initial offers are generated
        current_offers = {}
        for agent in sellers:
            current_offers[agent.agent_id] = np.random.normal(148, 5)
        for agent in buyers:
            current_offers[agent.agent_id] = np.random.normal(73, 5)

        # Loop over time steps
        i = 0
        while market_env.if_round_done is False:
            print(i, '-------')
            i += 1
            # Environment calculates what happens
            market_env.step(current_offers)
            # print(market_env.agents)

            # All agents receive observations from what environment generated
            for agent in sellers:
                agent.receive_observations_from_environment(market_env)
            for agent in buyers:
                agent.receive_observations_from_environment(market_env)

            # Clearing current offers
            current_offers.clear()

            # Agents who are not done yet decide on a new offer which are then inserted into the dictionary of current_offers
            for agent in sellers[market_env.not_done_sellers]:
                new_offer = agent.decide()
                current_offers[agent.agent_id] = new_offer
            for agent in buyers[market_env.not_done_buyers]:
                new_offer = agent.decide()
                current_offers[agent.agent_id] = new_offer

print(time.time() - start)
