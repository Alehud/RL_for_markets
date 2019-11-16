import numpy as np
import matplotlib.pyplot as plt


class Buyer:
    def __init__(self, budget: float, agent_id: int = None):
        self.budget = budget
        self.agent_id = agent_id
        self.price = None


class Seller:
    def __init__(self, prod_cost: float, agent_id: int = None):
        self.prod_cost = prod_cost
        self.id = agent_id
        self.price = None


buyers = [Buyer(200, i) for i in range(0, 100)]
sellers = [Seller(100, i) for i in range(0, 100)]

for buyer in buyers:
    buyer.price = 100 + np.random.exponential(5)
    if buyer.price > buyer.budget:
        buyer.price = buyer.budget
    # print(buyer.price)

for seller in sellers:
    seller.price = 200 - np.random.exponential(5)
    if seller.price < seller.prod_cost:
        seller.price = seller.prod_cost
    # print(seller.price)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 8), tight_layout=True)
ax[0].set_xlim(100, 170)
ax[1].set_xlim(130, 200)

while(True):
    data_buy = np.array([o.price for o in buyers])
    data_sell = np.array([o.price for o in sellers])

    # ax[0].cla()
    # ax[1].cla()

    _, _, bars0 = ax[0].hist(data_buy, 10, color='blue')
    _, _, bars1 = ax[1].hist(data_sell, 10, color='blue')
    plt.draw()
    plt.pause(0.1)
    _ = [b.remove() for b in bars0]
    _ = [b.remove() for b in bars1]

    for o in buyers:
        if np.all(data_sell > o.price):
            o.price += 1
        if o.price > o.budget:
            o.price = o.budget

    for o in sellers:
        if np.all(data_buy < o.price):
            o.price -= 1
        if o.price < o.prod_cost:
            o.price = o.prod_cost


plt.show()

# data = sorted(200 - np.random.exponential(5, 10000), reverse=True)
# print(data)
# fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
# ax.hist(data, 50)
# plt.show()




