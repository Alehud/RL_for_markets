import numpy as np
import matplotlib.pyplot as plt

# num = 10
# games = 30
# y = np.zeros((2*num, games))
# for i in range(num):
#     with open('nonlinear_blackbox_agent_reward_aggro_0.0_buyer_' + str(i) + '.txt', 'r') as f:
#         y[i] = np.fromstring(f.read(), dtype=float, sep=' ')
# for i in range(num):
#     with open('nonlinear_blackbox_agent_reward_aggro_0.0_seller_' + str(i) + '.txt', 'r') as f:
#         y[i+num] = np.fromstring(f.read(), dtype=float, sep=' ')
#
# x = np.linspace(1, 30, num=30)
#
# plt.scatter(x, y[4])
# plt.show()
aggro = 0.4
rewards_buyers = np.load('rewards_buyers_aggro' + str(aggro) + '.npy')
rewards_sellers = np.load('rewards_sellers_aggro' + str(aggro) + '.npy')
demands_agents = np.load('demands_agents_aggro' + str(aggro) + '.npy')
demand_aggro = np.load('demands_aggro_aggro' + str(aggro) + '.npy')
print(demands_agents)
print(demand_aggro)

fig, ax = plt.subplots(figsize=(14, 8), tight_layout=True)
# ax.set_title(f'{lls_to_show} Landau levels (out of ${nn}$); $n = {n}$; $C={c_val}$', fontsize=20)
# ax.set_title(f'$C={c_val}; N={n}$', fontsize=20)
# ax.set_ylabel('$E$', fontsize=20)
# ax.set_xlabel('$k_z$', fontsize=20)
ax.set_ylim([0, 60])
ax.scatter(np.arange(np.size(demands_agents)), demands_agents)
ax.scatter(np.arange(np.size(demand_aggro)), demand_aggro)
plt.show()





