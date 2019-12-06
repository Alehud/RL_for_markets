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
You will also have to install the internal libarary lib. 
For this in the root directory run:
~~~
pip3 install -e lib
~~~
##Library description
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

For example files 'test_....py' are simulating the games for stated type of agents without
reinforcement learning. Files 'rl_script....py' and 'DDPG_experiments....ipynb' are simulating the games for agents with
reinforcement learning.

Results of simulations are stored in the folder 'results'.

## Example
For example, you can first try running the file 'test_nonlinear_blackbox_agent.py' 
(you can play with parameters defined at the beginning of the file). Then run the file 
'plot_nonlinear_blackbox_agent.py', which will plot the results of the simulations.

As another example, you could run 'test_linear_generic_agent.py' and see in real time
 the distribution of offers. You will observe how buyers and sellers linearly adjust their
 offers until they meet in the middle and make deals with each other.
 
A lot of other simulations could be done. However, most probably they will require
 from the user some simple implementation of plotting or data saving to observe the results.


