import pandas as pd
import numpy as np
from agents import Buyer, Seller
from environments import MarketEnvironment
from info_settings import BlackBoxSetting, SameSideSetting, OtherSideSetting, BothSidesSetting, DealInformationSetting, FullInformationSetting
from matchers import RandomMatcher
import warnings
# pandas setting warnings can be ignored, as it is intendend often
warnings.simplefilter("ignore")


john = Seller('Seller John', 100)
nick = Seller('Seller Nick', 90)
alex = Buyer('Buyer Alex', 130)
kevin = Buyer('Buyer Kevin', 110)
sellers = [john, nick]
buyers = [alex, kevin]

market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_steps=30,
                               matcher=RandomMatcher(reward_on_reference=True), setting=SameSideSetting)

init_observation = market_env.reset()

step1_offers = {
    'Buyer Alex': 120,
    'Buyer Kevin': 105,
    'Seller John': 110,
    'Seller Nick': 105
}
# print(step1_offers)
observations, rewards, done, _ = market_env.step(step1_offers)
# print(pd.DataFrame(market_env.deal_history))
print(observations)
# print(rewards)
# print(done)
# print(market_env.offers)
# print(market_env.realized_deals)


step2_offers = {
    'Buyer Kevin': 95,
    'Seller John': 115,
}
# print(step2_offers)
observations, rewards, done, _ = market_env.step(step2_offers)
# print(pd.DataFrame(market_env.deal_history))
print(observations)
# print(rewards)
# print(done)
# print(market_env.offers)
# print(market_env.realized_deals)

step3_offers = {
    'Buyer Kevin': 105,
    'Seller John': 100,
}
# print(step3_offers)
observations, rewards, done, _ = market_env.step(step3_offers)
# print(pd.DataFrame(market_env.deal_history))
print(observations)
# print(rewards)
# print(done)
# print(market_env.offers)
# print(market_env.realized_deals)

# print(market_env.sellers)
# print(market_env.agents)
# print(market_env.agents['id'])
# print(market_env.agent_ids)
# print(market_env.agent_roles)
# print(market_env.max_steps)
# print(market_env.matcher)
# print(market_env.setting)
# print(market_env.n_sellers)
# print(market_env.matched)
# print(market_env.deal_history)
# print(market_env.offers)
# print(market_env.current_actions)
# print(market_env.realized_deals)
# print(market_env.time)
# print(market_env.done)


