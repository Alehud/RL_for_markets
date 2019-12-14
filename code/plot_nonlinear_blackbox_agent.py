import numpy as np
import matplotlib.pyplot as plt

aggro = 0.5
rewards_buyers = np.load('results/aggressive agent/rewards_buyers_aggro' + str(aggro) + '.npy')
rewards_sellers = np.load('results/aggressive agent/rewards_sellers_aggro' + str(aggro) + '.npy')
demands_agents = np.load('results/aggressive agent/demands_agents_aggro' + str(aggro) + '.npy')
demand_aggro = np.load('results/aggressive agent/demands_aggro_aggro' + str(aggro) + '.npy')

fig, ax = plt.subplots(figsize=(14, 8), tight_layout=True)
ax.set_ylabel('demand', fontsize=22)
ax.set_xlabel('time step', fontsize=22)
ax.set_ylim([0, 60])
ax.scatter(np.arange(np.size(demands_agents)), demands_agents)
ax.scatter(np.arange(np.size(demand_aggro)), demand_aggro)
plt.show()





