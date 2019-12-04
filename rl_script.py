#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import doubleauction


# In[2]:


import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=64, hidden2=64):
        super(Critic, self).__init__()
        self.bn1 = nn.BatchNorm1d(nb_states)
        self.fc1 = nn.Linear(nb_states, hidden1)
        
        self.bn2 = nn.BatchNorm1d(hidden1+nb_actions)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        
        self.bn3 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, s, a):
        out = self.fc1(self.bn1(s))
        out = self.relu(out)

        out = self.fc2(self.bn2(torch.cat([out,a],1)))
        out = self.relu(out)
        out = self.fc3(self.bn3(out))
        return out


# In[3]:



class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=64, hidden2=64, init_w=3e-3):
        super(Actor, self).__init__()
        self.bn1 = nn.BatchNorm1d(nb_states)
        self.fc1 = nn.Linear(nb_states, hidden1)
        
        self.bn2 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        
        self.bn3 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        
        self.relu = nn.ReLU()   
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        out = self.fc1(self.bn1(x))
        out = self.relu(out)
        out = self.fc2(self.bn2(out))
        out = self.relu(out)
        out = self.fc3(self.bn3(out))
        out = self.softplus(out)
        
        return out
    
    


# In[18]:



import scipy.stats
import numpy as np
from doubleauction.agents import Buyer, Seller
import torch
        
class DDPGSeller(Seller):
    def __init__(self, agent_id: str,  reservation_price: float, *,
                    max_noise=50., anneal_steps=1e4, min_noise=5.,
                    batch_size = 64, mem_size=100000, lr=1e-2, width_actor=64, width_critic=64,
                    tau=1e-2, discount=0.98, wd = 1e-4):
        """
        A seller agent that extends the market agent
        :param agent_id: a unique id that differentiates this agent to other agents
        :param reservation_price: the reservation price, or minimum price that this agent is
        willing to sell
        """
        super().__init__(agent_id, reservation_price)
        self.max_noise = max_noise
        self.anneal_steps = anneal_steps
        self.min_noise = min_noise
        
        self.sigma = max_noise
        
        self.actor = Actor(5, 1, hidden1=width_actor, hidden2=width_actor)
        self.actor_target = Actor(5, 1, hidden1=width_actor, hidden2=width_actor)
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(), lr=1e-2, weight_decay=wd)

        self.critic = Critic(5, 1, hidden1=width_critic, hidden2=width_critic)
        self.critic_target = Critic(5, 1, hidden1=width_critic, hidden2=width_critic)
        self.critic_optim  = torch.optim.Adam(self.critic.parameters(), lr=1e-2, weight_decay=wd)

        self.memory = SequentialMemory(limit=10000, window_length=1)
    
        self.tau = tau
        self.lr = lr
        self.width_actor = width_actor
        self.width_critic = width_critic

        self.mem_size = mem_size
        self.batch_size = batch_size
        
        self.discount = discount
        
        self.criterion = nn.MSELoss()
        
        self.eval = False
        
        self.game_count = 0
        
        self.new_game()
        

    
    def new_game(self):
        self.last_demand = 0
        self.game_first = True
        self.round_first = True
        self.last_successful = False
        
        self.state = None
        self.action = None
        
        self.new_round()
        
        if not self.eval:
            self.sigma = self.max_noise - self.game_count * (self.max_noise - self.min_noise) / self.anneal_steps
            self.sigma = max(self.min_noise, self.sigma)
            
        self.game_count += 1
        
    def new_round(self):
        self.round_first = True
        
        
    def decide(self, observations):
        
        ## update state
        self.state = np.array([self.last_demand, self.last_successful,
                               self.game_first, self.round_first,
                               self.reservation_price], dtype=np.float)
        s = torch.tensor(self.state).float().unsqueeze(0)

        with torch.no_grad():
            self.actor.eval()
            if self.eval:
                a = self.actor(s)
                a = a.item()
            else:
                
                n = np.random.normal(0, self.sigma)

                a = self.actor(s).item() + n
                
                a = abs(a)
                
                
                
            self.actor.train()

        self.action = a
        
        self.last_demand = a
        self.game_first = False
        self.round_first = False

        return self.reservation_price + a
  

    def observe(self, reward, done):
        self.memory.append(self.state, self.action, reward, done)
        self.last_successful = (reward > 0)

        
    def learn(self):
        
        # perform grad descent    
        
        # Sample batch
        state_batch, action_batch, reward_batch,         next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target(
                torch.tensor(next_state_batch, dtype=torch.float),
                self.actor_target(torch.tensor(next_state_batch, dtype=torch.float)))

            target_q_batch = torch.tensor(reward_batch, dtype=torch.float) +                 self.discount*torch.tensor(terminal_batch, dtype=torch.float)*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic(torch.tensor(state_batch, dtype=torch.float), 
                                torch.tensor(action_batch, dtype=torch.float))

        value_loss = self.criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic(
            torch.tensor(state_batch, dtype=torch.float),
            self.actor(torch.tensor(state_batch, dtype=torch.float)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        


# # Create environment

# In[19]:


import doubleauction
from doubleauction.agents import RandomSeller, RandomBuyer
from doubleauction.environments import MarketEnvironment
from doubleauction.matchers import RandomMatcher
from doubleauction.util import SequentialMemory, hard_update, soft_update
from doubleauction.util import generate_seller_prices_paper, generate_buyer_prices_paper


# In[20]:


records = {}
records['rewards'] = []
records['demands'] = []
records['prices'] = []


# In[21]:


rewards = []
epochs = 10
warmup_epochs = 5
seller_agent = DDPGSeller('learner', 0, 
                          discount = 0.98, lr = 1e-3, max_noise=50., min_noise=5., anneal_steps=300,
                          wd = 1e-4, mem_size=1e6)


# In[23]:


for e in range(epochs):
    seller_agent.reservation_price = generate_seller_prices_paper(1)[0]
    
    sellers = []
    for ii, p in enumerate(generate_seller_prices_paper(19)):
        sellers.append(RandomSeller('s'+str(ii), p))
    sellers.append(seller_agent)

    buyers = []
    for ii, p in enumerate(generate_buyer_prices_paper(20)):
        buyers.append(RandomBuyer('b'+str(ii), p))

    agents = sellers + buyers
    
    seller_agent.new_game()
    
    setting = {
        'self_last_offer': False,
        'same_side_last_offers': False,
        'other_side_last_offers': False,
        'completed_deals': False,
        'current_time': False,
        'max_time': False,
        'n_sellers': False,
        'n_buyers': False
    }

    market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_steps=10,
                                   matcher=RandomMatcher(reward_on_reference=True), setting=setting)
    init_observation = market_env.reset()

    round_avg = 0.
    offer_avg = 0.
    time_avg = 0.
    
    records['demands'].append([])
    records['rewards'].append([])
    records['prices'].append(seller_agent.reservation_price)

    for n_round in range(10):
        
        init_observation = market_env.reset()
        observations = {k.agent_id:None for k in agents}
        done = {k.agent_id:False for k in agents}
        reward_hist = []
        rounds = 0
        terminate_round = False
        
        seller_agent.new_round()
        
        records['demands'][-1].append([])
        records['rewards'][-1].append([])
        
        offers_list = []
        
        while not terminate_round:
            offers = {}

            offers = {a.agent_id : a.decide(observations[a.agent_id]) for a in agents}

            observations, rewards, done, _ = market_env.step(offers)
            reward_hist.append(rewards)
            rounds += 1

            terminate_round = all(done.values()) or rounds >= 10 or done['learner']

            # create record of experience
            seller_agent.observe(rewards['learner'], terminate_round)
            
            offers_list.append(offers['learner'] - seller_agent.reservation_price)
            
            records['demands'][-1][-1].append(offers['learner'] - seller_agent.reservation_price)
            records['rewards'][-1][-1].append(rewards['learner'])
            
            round_avg += rewards['learner']

            time_avg += 1
    
        offer_avg += sum(offers_list) / len(offers_list)
#         time_vs_rewards.append(round_avg)
#         time_vs_demand.append(sum(offers_list) / len(offers_list))
        
        if e >= warmup_epochs:
            seller_agent.learn()
    
    print('Epoch: {}, Avg. earnings: {}, Avg. demand: {}, Avg. time: {}'.format(e, round_avg / 10., 
                                                                            offer_avg / 10.,
                                                                            time_avg / 10.))
    
    


# In[27]:


torch.save(records, 'results/records1')

torch.save({'actor':seller_agent.actor.state_dict(),
           'actor_target':seller_agent.actor_target.state_dict(),
          'critic':seller_agent.critic.state_dict(),
          'critic_target':seller_agent.critic_target.state_dict()}, 'results/models1')

