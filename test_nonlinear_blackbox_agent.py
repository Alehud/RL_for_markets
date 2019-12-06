import numpy as np
from doubleauction.environments import MarketEnvironment
from doubleauction.matchers import RandomMatcher
from doubleauction.util import generate_buyer_prices_paper, generate_seller_prices_paper
from doubleauction.agents.nonlinear_blackbox_agent import NonlinearBlackBoxBuyer, NonlinearBlackBoxSeller
import matplotlib.pyplot as plt
import os
import warnings
# pandas setting warnings can be ignored, as it is intendend often
warnings.simplefilter("ignore")
my_path = os.getcwd()


# Define the initial number of agents
n_sellers = 100
n_buyers = 100
n_game = 30
n_round = 20
num = 10

# Create initial agents with names and reservation prices
# All agents are the same for now
res_prices = generate_seller_prices_paper(discrete=False, count=n_sellers)
names = ['Seller ' + str(i) for i in range(1, n_sellers + 1)]
sellers = np.array([NonlinearBlackBoxSeller(agent_id=names[i], reservation_price=res_prices[i]) for i in range(n_sellers)])
res_prices = generate_buyer_prices_paper(discrete=False, count=n_buyers)
names = ['Buyer ' + str(i) for i in range(1, n_buyers + 1)]
buyers = np.array([NonlinearBlackBoxBuyer(agent_id=names[i], reservation_price=res_prices[i]) for i in range(n_buyers)])


# For plotting
# fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
# ax.set_xlim(75, 150)

rewards_buyers = np.zeros((num, n_game))
rewards_sellers = np.zeros((num, n_game))
print(rewards_buyers)

aggro_array = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
for aggro in aggro_array:
    print("AGGRO", aggro, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # Loop over games
    for g in range(n_game):
        print("GAME", g, '=================================================================================================================')

        # Define parameters of each round
        max_time = 50
        matcher = RandomMatcher(reward_on_reference=True)

        # Create market environment
        market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_time=max_time, matcher=matcher)

        # HERE AGENTS LEARN AND ADJUST THEIR COEFS (for now the are constant)
        for agent in sellers:
            agent.coefs = np.array([1-0.75497335, 0.75497335, aggro])
        for agent in buyers:
            agent.coefs = np.array([1-0.75497335, 0.75497335, aggro])

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
                current_offers[agent.agent_id] = np.random.normal(148, 5)
            for agent in buyers:
                current_offers[agent.agent_id] = np.random.normal(73, 5)

            # Loop over time steps
            i = 0
            while market_env.if_round_done is False:
                # print(i, '-------')
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

        for i in range(num):
            rewards_sellers[i][g] = sellers[i].reward
            rewards_buyers[i][g] = buyers[i].reward
                # for plotting
                # _, _, bars0 = ax.hist(list(current_offers.values()), 50, color='blue')
                # plt.draw()
                # plt.pause(0.1)
                # _ = [b.remove() for b in bars0]

    for n in range(num):
        reward_buyers_per_game = rewards_buyers[n]
        reward_sellers_per_game = rewards_sellers[n]
        sellerfname = 'nonlinear_blackbox_agent_reward_aggro_' + str(aggro) + '_seller_' + str(n) + '.txt'
        buyerfname = 'nonlinear_blackbox_agent_reward_aggro_' + str(aggro) + '_buyer_' + str(n) + '.txt'
        filename_s = os.path.join(my_path, sellerfname)
        filename_b = os.path.join(my_path, buyerfname)
        np.savetxt(filename_s, reward_sellers_per_game)
        np.savetxt(filename_b, reward_buyers_per_game)
