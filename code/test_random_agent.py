import numpy as np
from doubleauction.environments import MarketEnvironment
from doubleauction.matchers import RandomMatcher
from doubleauction.agents.random_agent import RandomBuyer, RandomSeller
import matplotlib.pyplot as plt
import warnings
# pandas setting warnings can be ignored, as it is intendend often
warnings.simplefilter("ignore")
import time

start = time.time()


# Define the initial number of agents, the number of rounds and games
n_sellers = 100
n_buyers = 100
n_game = 1
n_round = 10

# Create initial agents with names and reservation prices
# All agents are the same for now
res_prices = np.random.normal(100, 5, n_sellers)
names = ['Seller ' + str(i) for i in range(1, n_sellers + 1)]
sellers = np.array([RandomSeller(agent_id=names[i], reservation_price=res_prices[i]) for i in range(n_sellers)])
res_prices = np.random.normal(200, 5, n_buyers)
names = ['Buyer ' + str(i) for i in range(1, n_buyers + 1)]
buyers = np.array([RandomBuyer(agent_id=names[i], reservation_price=res_prices[i]) for i in range(n_buyers)])


# For plotting
# fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
# ax.set_xlim(95, 205)

# Loop over games
for g in range(n_game):
    print("GAME", g, '=================================================================================================================')

    # Define parameters of each round
    max_time = 50
    matcher = RandomMatcher(reward_on_reference=True)

    # Create market environment
    market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_time=max_time, matcher=matcher)

    # HERE AGENTS LEARN AND ADJUST THEIR COEFS (random agents don't need any coefficients, they don't learn)

    # Reset agents' rewards and observations
    for agent in sellers:
        agent.reward = 0.0
        agent.observations = {}
    for agent in buyers:
        agent.reward = 0.0
        agent.observations = {}

    # Loop over rounds
    for r in range(n_round):
        print("ROUND", r, '-----------------------------------------------')

        # Reset market environment
        market_env.reset()

        # Initial offers are generated
        current_offers = {}
        for agent in sellers:
            current_offers[agent.agent_id] = np.random.normal(200, 5)
        for agent in buyers:
            current_offers[agent.agent_id] = np.random.normal(100, 5)

        # Loop over time steps
        i = 0
        while market_env.if_round_done is False:
            print(i, '-------')
            i += 1
            # Environment calculates what happens
            market_env.step(current_offers)

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

            # for plotting
            # _, _, bars0 = ax.hist(list(current_offers.values()), 50, color='blue')
            # plt.draw()
            # plt.pause(0.1)
            # _ = [b.remove() for b in bars0]

stop = time.time()
print(stop-start)
