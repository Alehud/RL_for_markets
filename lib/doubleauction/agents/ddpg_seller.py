from .agents import Buyer, Seller
import scipy.stats
import numpy as np
from ..models import Actor
from doubleauction.util import SequentialMemory, hard_update, soft_update
import torch
        
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
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=wd)

        self.critic = Critic(5, 1, hidden1=width_critic, hidden2=width_critic)
        self.critic_target = Critic(5, 1, hidden1=width_critic, hidden2=width_critic)
        self.critic_optim  = torch.optim.Adam(self.critic.parameters(), lr=lr, weight_decay=wd)

        self.memory = SequentialMemory(limit=mem_size, window_length=1)
    
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
                d = self.actor(s).item()
                a = d + n

#                 a = abs(a)
                print(d)
                
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
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target(
                torch.tensor(next_state_batch, dtype=torch.float),
                self.actor_target(torch.tensor(next_state_batch, dtype=torch.float)))

            target_q_batch = torch.tensor(reward_batch, dtype=torch.float) + \
                self.discount*torch.tensor(terminal_batch, dtype=torch.float)*next_q_values

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
        