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

