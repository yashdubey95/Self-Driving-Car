# Self-Driving-Car

This repository contains code for the PyTorch implementation of a Self Driving Car from the online course "Artificial Intelligence A-Z™: Learn How To Build An AI" provided by Udemy and SuperDataScience and taught by Hadelin de Ponteves and Kirill Eremenko.

## Introduction:

The goal of this project is to design a modelled-version of a self-driving car that can learn how to navigate itself between two points in an environment by using the concept of Deep-Q Learning.

In order to understand the concept of Deep-Q Learning we have to understand the foundation behind one of the most popular Reinforcement Learning algorithm "Q-Learning".

### Reinforcement Learning:
Reinforcement Learning(RL) is a type of machine learning technique that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences. RL can be referred to as a type of unsupervised machine learning algorithm, as the agent is expected to learn by being set loose in its environment, without the need for specific training data to be generated and then used to teach the agent.

Goal of Reinforcement Learning:
Given the current state we are in, choose the optimal action which will maximise the long-term expected reward provided by the environment.
Following are the steps in which an RL agent learns:
  1. The agent observes an input state.
  2. An action is determined by a decision making function (policy).
  3. The action is performed.
  4. The agent receives a scalar reward or reinforcement from the environment.
  5. Information about the reward given for that state / action pair is recorded.

By performing actions, and observing the resulting reward, the policy used to determine the best action for a state can be fine-tuned. Eventually, if enough states are observed an optimal decision policy will be generated and we will have an agent that performs perfectly in that particular environment.

One of the interesting problems that arises when using Reinforcement Learning is the tradeoff between exploration and exploitation. If an agent has tried a certain action in the past and got a decent reward, then repeating this action is going to reproduce the reward. In doing so, the agent is exploiting what it knows to receive a reward. On the other hand, trying other possibilities may produce a better reward, so exploring is definitely a good tactic sometimes. Without a balance of both exploration and exploitation the RL agent will not learn successfully. The most common way to achieve a nice balance is to try a variety of actions while progessively favouring those that stand out as producing the most reward.

### Markov Decision Process:
As stated above the main goal of Reinforcement Learning is to perform a sequence of steps in order to maximize the reward. These sequences can be seen as a series of consequences to the actions taken in a particular state, and thus hold an active part in choosing the agent's next move.
As the agent explores the environment further and further, the amount of information it holds and needs to process also increases, and it might reach to a point where it is infeasible to carry out calculations. Thus, to tackle this problem we assume that each state follows the Markov Property i.e. The result of a future state (including your choice and the environment) depends on the state that you are in now and not on how you got to that state.

### Bellman Equation:
The Bellman Equation is at the core of the Q-learning algorithm and one of the key equations in the world of reinforcement learning as it helps in finding the optimal policy and value functions. 
The Bellman Equation can solve the Markov Decision Process, also since it is recursive in nature it starts from the max reward and works it way backward and thus allows for the rewards from the future states to propagate far-off past states.
The Bellman Equation for a deterministic environment states that the value of a given state "V(s)" is equal to the max action (action which maximises the value) of the sum of the reward of the optimal action in the given state "R(s,a)" and product of the discount factor "γ"(diminishes the reward over time) and the next state’s value V(s').

![alt text](https://github.com/yashdubey95/Self-Driving-Car/blob/master/images/BEDE.png)

For a stochastic environment the Bellman Equation for deterministic environment needs to be modified a little bit in order to factor in the uncertainty as taking an action in a state does not gaurantee that the agent will end up in the intended next state. A variable P(s,a,s') that gives the probability of getting into state s' as a result of an action a while being in state s, summed over the total number of future states. The Bellman Equation for stochastic environment is given below:

![alt text](https://github.com/yashdubey95/Self-Driving-Car/blob/master/images/BESE.png)

* Example of Optimal Value V(s) for Bellman Equation:-

![alt text](https://github.com/yashdubey95/Self-Driving-Car/blob/master/images/BEST.png)

* Example of Optimal Policy for Bellman Equation:-

![alt text](https://github.com/yashdubey95/Self-Driving-Car/blob/master/images/BEPD.png)


### Q-Learning:
The Q-learning algorithm builts on the foundation set by the Bellman Equation to create a table "Q-table"containing all the possible state-action combinations with their respective Q-values "Q(s,a)" values which quantify an action taken in a particular state. The Q-values basically tell the agent which action from a set of actions will be the most beneficial to be taken from a particular state, rather than looking at the value of the state to decide its next move.

* Q-learning update rule for Deterministic Environment:-

![alt text](https://github.com/yashdubey95/Self-Driving-Car/blob/master/images/QLDE.png)

* Q-learning update rule for Stochastic Environment:-

![alt text](https://github.com/yashdubey95/Self-Driving-Car/blob/master/images/QLSE.png)

* Bellman Equation v/s Q-Learning:-

![alt text](https://github.com/yashdubey95/Self-Driving-Car/blob/master/images/BLvsQL.png)

Pseudocode of  Q-learning:
	1. Initialize the Values table "Q(s, a)" randomly.
  2. Observe the current state "s".
  3. Choose an action "a" for that state based on one of the action selection policies (eg. epsilon-greedy).
  4. Take the action, and observe the reward  "r" as well as the new state  "s".
  5. Update the Value for the state using the observed reward and the maximum reward possible for the next state. The updating is done according to the formula and parameters described above.
  6. Set the state to the new state, and repeat the process until a terminal state is reached.
  
### Deep Q-Learning:

Q-Learning is a powerful algorithm to train an agent through reinforcement learning, but it suffers from the problem of scalability as the number of states and actions become increasingly large, even a simple game of Tic Tac Toe has hundreds of different state-action combinations. To solve this we use a Deep Neural Network (hence the name "Deep Q-Learning") and replace it with the Q-table, this neural network is referred to as an approximator or the approximation function and its job is to approximate or predict those Q-values.

![alt text](https://github.com/yashdubey95/Self-Driving-Car/blob/master/images/QLvsDQL.PNG)

Here the cost function that we aim to minimize is the mean squared error between the predicted Q-value and the target Q-value, now since this is a reinforcement learning task initially there wont be a target/actual Q-value, but after a certain amount time ideally a batch size, we provide the agent with whatever it has learnt in the form of memory to train the network further, thus letting it learn from its experience in the environment, this concept of loading memory buffer after a certain amount of time is known as Experience Replay and it makes sure that the agent keeps on learning top of its new behaviour rather than learning things that are no longer relevant, for example teaching an agent how to run might follow this sequence learn to crawl => learn to walk => learn to run, here it wont make sense for the agent to learn how to crawl again after it has already learnt how to walk.

## Dependencies:

To install all the dependencies for this project use the following code:
```bash
conda install -c peterjc123 pytorch-cpu
conda install -c conda-forge kivy
conda install matplotlib
```

## Project Files:
1. car.kv: Contains the code for the GUI made with Kivy.
2. ai.py: Contains the PyTorch implementation for the Deep-Q Network architecture.
3. map.py: Main file that contains the code for the AI part of the project integrating the GUI and the Deep-Q Network.

## Usage:
To run the project simply run the map.py file using the following command:
```bash
python map.py
```

### Output:
![alt text](https://github.com/yashdubey95/yashdubey95.github.io/blob/master/assets/img/portfolio/portfolio-2.jpg)
