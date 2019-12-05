#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import doubleauction


# In[2]:


from doubleauction.util import OrnsteinUhlenbeckProcess


# In[3]:


import matplotlib.pyplot as plt
theta = 0.7
sigma = 15.
p = OrnsteinUhlenbeckProcess(theta=theta, sigma = sigma)
sigma * sigma / 2 / theta


# In[4]:


l = []
for i in range(100):
    l.append(p.sample())
    
plt.plot(l)


# # Create environment

# In[5]:


import matplotlib.pyplot as plt


# In[6]:


import doubleauction
from doubleauction.agents import RandomSeller, RandomBuyer, DDPGSellerOU
from doubleauction.environments import MarketEnvironment

from doubleauction.matchers import RandomMatcher

from doubleauction.util import SequentialMemory, hard_update, soft_update
from doubleauction.util import generate_seller_prices_paper, generate_buyer_prices_paper


# In[7]:


records = {}
records['rewards'] = []
records['demands'] = []
records['prices'] = []


# In[8]:


rewards = []
epochs = 500
warmup_epochs = 20
seller_agent = DDPGSellerOU('learner', 0, 
                                ou_theta=.7, ou_mu=.0, ou_sigma=15., sigma_min=3.5, anneal_steps=300*10*10,
                                  discount = 0.97, lr = 3e-4, 
                                  wd = 1e-4, mem_size=500000, tau=5e-3)


# # Run the training algorithm

# In[9]:


mdict = torch.load('results/models2')
seller_agent.actor.load_state_dict(mdict['actor'])
seller_agent.actor_target.load_state_dict(mdict['actor_target'])

seller_agent.critic.load_state_dict(mdict['critic'])
seller_agent.critic_target.load_state_dict(mdict['critic_target'])


# In[10]:


get_ipython().run_cell_magic('time', '', "for e in range(epochs):\n    seller_agent.reservation_price = generate_seller_prices_paper(1)[0]\n    \n    sellers = []\n    for ii, p in enumerate(generate_seller_prices_paper(19)):\n        sellers.append(RandomSeller('s'+str(ii), p))\n    sellers.append(seller_agent)\n\n    buyers = []\n    for ii, p in enumerate(generate_buyer_prices_paper(20)):\n        buyers.append(RandomBuyer('b'+str(ii), p))\n\n    agents = sellers + buyers\n    \n    seller_agent.new_game()\n    \n#     setting = {\n#         'self_last_offer': False,\n#         'same_side_last_offers': False,\n#         'other_side_last_offers': False,\n#         'completed_deals': False,\n#         'current_time': False,\n#         'max_time': False,\n#         'n_sellers': False,\n#         'n_buyers': False\n#     }\n    \n    ROUNDS_PER_GAME = 10\n\n    market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_time=10, ## not the same as rounds per game!!\n                                   matcher=RandomMatcher(reward_on_reference=True))\n    init_observation = market_env.reset()\n\n    round_avg = 0.\n    offer_avg = 0.\n    time_avg = 0.\n    \n    records['demands'].append([])\n    records['rewards'].append([])\n    records['prices'].append(seller_agent.reservation_price)\n\n    for n_round in range(10):\n        \n        init_observation = market_env.reset()\n        observations = {k.agent_id:None for k in agents}\n        done = {k.agent_id:False for k in agents}\n        reward_hist = []\n        rounds = 0\n        terminate_round = False\n        \n        seller_agent.new_round()\n        \n        records['demands'][-1].append([])\n        records['rewards'][-1].append([])\n        \n        offers_list = []\n        \n        while not terminate_round:\n            offers = {}\n\n            for iagent in agents:\n                iagent.receive_observations_from_environment(market_env)\n                \n            offers = {a.agent_id : a.decide() for a in agents}\n\n            market_env.step(offers)\n            \n            rewards = market_env.rewards\n            \n            reward_hist.append(rewards)\n            rounds += 1\n\n            terminate_round = market_env.if_round_done or \\\n                                market_env.agents[market_env.agents['id'] == 'learner']['done'].iloc[0]\n\n            # create record of experience\n            seller_agent.memorize(rewards['learner'], terminate_round)\n            \n            offers_list.append(offers['learner'] - seller_agent.reservation_price)\n            \n            records['demands'][-1][-1].append(offers['learner'] - seller_agent.reservation_price)\n            records['rewards'][-1][-1].append(rewards['learner'])\n            \n            round_avg += rewards['learner']\n\n            time_avg += 1\n    \n        offer_avg += sum(offers_list) / len(offers_list)\n#         time_vs_rewards.append(round_avg)\n#         time_vs_demand.append(sum(offers_list) / len(offers_list))\n        \n        if e >= warmup_epochs:\n            seller_agent.learn()\n    \n    print('Epoch: {}, Avg. earnings: {}, Avg. demand: {}, Avg. time: {}'.format(e, round_avg / 10., \n                                                                            offer_avg / 10.,\n                                                                            time_avg / 10.))\n    \n    if (e + 1) % 100 == 0:\n        torch.save({'actor':seller_agent.actor.state_dict(),\n                   'actor_target':seller_agent.actor_target.state_dict(),\n                  'critic':seller_agent.critic.state_dict(),\n                  'critic_target':seller_agent.critic_target.state_dict()}, 'results/models_ou1_e{}'.format(e))")


# In[11]:


flatten = lambda l: [item for sublist in l for item in sublist]

l2 = flatten( records['rewards'] )
l3 = [sum(ll) for ll in l2]

plt.figure()
plt.plot(l3)
# plt.plot(smooth(l2, 10))


# In[12]:


torch.save(records, 'results/records_ou1')

torch.save({'actor':seller_agent.actor.state_dict(),
           'actor_target':seller_agent.actor_target.state_dict(),
          'critic':seller_agent.critic.state_dict(),
          'critic_target':seller_agent.critic_target.state_dict()}, 'results/models_ou1')


# In[13]:


torch.save(seller_agent.memory, 'results/memory_ou1')

