
# GESS project


## Installing the library

In the root directory, run:
~~~
pip3 install -e lib
~~~
You can then import from python with
```python
import doubleauction
```

## Plan of the simulation
~~~
- define the initial number of n_sellers, n_buyers
- create agents with their unique settings (type of the agent, agent_id, reservation price, info_setting)
loop over games:
    - make changes to agents, create/kill agents, update n_sellers, n_buyers
    - define parameters of the round (max_time, matcher) and create an environment
    - each agent determines the length of the coefs vector he wants as an input
    - for each agent a coefs vector is generated using the information about its length,
      n_sellers, n_buyers, max_time, and the previous coefs of this agent
    - rewards are reset
    loop over rounds:
        - initial offers are generated
        loop over time steps:
            - using current offers, environment calculates what happens and generates some observations
            - each agent receives receives some part of these observations depending oh his/her info_setting
            - each agent generates his/her new offer using observations received and coefs
            - new offers of all agents are composed into a list of current offers
        - data about the current round is collected and averaged
        - reward is accumulated
    - data about the current game is collected and averaged
        
        
~~~   
