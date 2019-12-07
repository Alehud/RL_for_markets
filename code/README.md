## Installing the library
For running the code you will need python of version not less than 3.7 and
the following libraries installed:
- numpy
~~~
pip install numpy
~~~
- scipy 
~~~
pip install scipy
~~~
- jupyter
~~~
pip install jupyter
~~~
- pandas 
~~~
pip install pandas
~~~
- matplotlib 
~~~
pip install matplotlib
~~~
- pytorch (follow the instructions at https://pytorch.org/get-started/locally/ )
- gym 
~~~
pip install gym
~~~
You will also have to install the internal utility library lib. For this in the git root directory run:
~~~
pip3 install -e code/lib
~~~


## Minimal reproducible code

The RL training code typically take up to 2 hours.
However for a minimal running example with pretrained parameters, open the jupyter notebook `code/minimal_ddpg_simulation.ipynb`. To do this, open a terminal and type
~~~
jupyter notebook
~~~
This should automatically open a browser window, if not, navigate to `localhost:8888` in a browser.
Navigate to and open `code/minimal_ddpg_simulation.ipynb`.
Run the jupyter notebook by clicking Cell > Run All. This should replicate the one of the core results of the project.

For minimal running examples without reinforcement learning, run `code/test_nonlinear_blackbox_agent.py`.
Then run `code/plot_nonlinear_blackbox_agent.py`, which will plot the results. You can also run
`code/test_linear_generic_agent.py` and observe in real time how buyers and sellers linearly
adjust their offers until they meet in the middle and make deals with each other.

## Library description
The main part of the library lib is in folder 'doubleauction'. It contains:
- Folder 'agents': agent classes of different types
- Folder 'environments': market environment class, it stores the data about 
the game and all agents taking part in it, and processes what happens during the game
- Folder 'matchers': matcher class, used by environment to match buyer and seller after
 they agree on the deal, computes the deal price
- Folder 'models': actor and critic classes, used for DDPG reinforcement learning algorithm
- Folder 'util': utility functions

## Project files description
Most of the .py and .ipynb files, which are not in the library are used to run simulations
of games for different types of agents.

For example files `test_*.py` are simulating the games for stated type of agents without
reinforcement learning. Files `rl_script*.py` and `DDPG_experiments*.ipynb` are simulating the games for agents with
reinforcement learning.

Results of simulations are stored in the folder 'results'.


