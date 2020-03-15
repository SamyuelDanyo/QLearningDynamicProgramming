# QLearningDynamicProgramming
Q-Learning and Dynamic Programming for World Grid Navigation. Reinforcement Learning experiment with insights into policy learning and hyper parameter tuning. Production QLearningDynamicProgramming module.

__For usage instructions please check [Usage README](https://github.com/SamyuelDanyo/QLearningDynamicProgramming/blob/master/docs/README.txt)__

__For full documentation - system design, experiments & findings please read [QLearningDynamicProgrammingDoc Report](https://github.com/SamyuelDanyo/QLearningDynamicProgramming/blob/master/docs/QLearningDynamicProgrammingDoc.pdf)__

__[GutHub Pages Format](https://samyueldanyo.github.io/q-learning-dynamic-programming/)__

## Introduction
In this report I present my implementation of the Q-Learning (**QL**) algorithm, which is from the family of Reinforcement Learning (**RL**) algorithms. The implementation is specifically designed for solving a world grid navigation problem, utilizing either QL or Dynamic Programming (**DP**). The method is a full-custom implementation in **Python**. Additionally, visualizations, performance results and analysis of a world grid navigation experiment are discussed. 

__*The goal of the project is to study the effectiveness of QL to dynamically learn, as well as, the effects of the system’s configuration & the input reward topology on performance.*__

![Q-Learning Optimal Policy](/res/opt_pol_traj.PNG)

The target world grid navigation problem to is defined as follows: The world is a 10x10 grid, with 100 states total. The agent starts at state 0 (the idle state), attempting to reach state 100 (the goal state). In order to guide the agent, a reward is given based on its state and action. The reward is loaded from task1.mat, and can be seen in the exported jupyter notebook (**QLearningDynamicProgramming.html**).

Four different parameter decay functions are implemented, namely: k_decay, const_k_decay, log_k_decay, const_log_k_decay.
For performance analysis of the learnt Q-Function, a few different metrics are observed: success in deducting an optimal policy for reaching the goal state (success run rate), training time, success in reaching the goal state during learning (success trial rate), optimal policy reward, optimal policy number of actions, exploration/exploitation ratio.

While the main metrics for evaluating the performance of the method are: the  successful learning of the optimal Q-Function distribution, as well as, optimal policy reward & action sequence length, observing execution time & exploration/exploitation ratio gives interesting insights into how the learning parameters’ configuration affects the learning process and hence how is that reflected into the resultant performance.

These effects are studied through running the QL algorithm with various learning parameters configurations, mixing the above-mentioned epsilon/learning rate decay functions with reward discount factors in [0.5;0.9]. Each configuration was run 10 times in order for its success rate in extracting an optimal policy to be found.

The exploration/exploitation ratio can be directly linked to the ability of the configuration to yield an optimal policy. The effects of the decay rate and reward discounts on the ratio are also analyzed.

The algorithm is implemented mainly through using NumPy for matrices manipulation and calculations. Helper functions are implemented for parameter decay, evaluation metrics extraction, as well as, wrapped training and displaying results using matplotlib for plotting graphs and some other basic helper packages.

## Environment Class
### Design
The class implements the environment functionality in the QL algorithm. It is responsible for tracking the agent location (state) and performing the agent steps, by computing the next state and issuing reward.

## Agent Class
### Design
The class implements the agent functionality in the QL algorithm. It is responsible for making decisions (taking actions), utilizing the epsilon-greedy approach, based on epsilon, current Q-Function and current state (valid actions).

## Q-Learning Class
### Design
Q-Learning is an unsupervised Machine Learning approach from the family of Reinforcement Learning algorithms. It deals with uncertain and usually unknown system by utilizing trial and error through exploration and incentive in the form of reward. 

The QL system is comprised of an environment and an agent. The objective is for the agent to maximize the received reward while performing a given task.

Q-Learning is based on a few ideas:

  __Markov Decision Process (MDP):__ is a model for a dynamic system which exhibits the so-called Markov property: [given the present, the future does not depend on the past.]
  
  __MDP__ comprises of a few elements:
  
    1. Set of states:  S = {s1; s2: s}
    2. Set of actions: A = {a1; a2: a}
    3. (State) Transition Function: 
      •	Deterministic – f: S x A => S
      •	Non-deterministic (Probability of the system reaching the new state s' after taking action a at state s) – f: S x A x S => [0:1]
    4. Reward & Return: When the agent takes an action a to make a transition from state s to next state s', reward r is assigned by the environment. The agent receives the reward in the next state s'.
      •	Reward Function: S x A x S => R
      •	Total Reward (Return): The total reward expected in the future by doing k-steps (transitions), with future rewards being discounted by a factor of y.
        - R0 = r1 + y*r2 + y^2*r3 ... + y^(k-1)*rk
  
  __Controlling MDP__ by imposing a decision-making mechanism - __Policy__:

    1. Deterministic (Based on a state, an action is taken).
      •	n: S => A
    2. Non-deterministic (Based on a state and action a probability is given).
      •	n: S x A => [0:1]
  
  __Q-Function (QF):__ Measures the “worth” of taking a specific action __a__ at a particular state __s__ under a given policy __n__ based on the __expected Return__.
  
    1. Qn: S x A => R
  
  __Optimal Policy:__ Maximizes (with respect to a given task) the __QF__ over all possible (__s,a__) pairs.
  
  __Bellman Equation:__ Provides an iterative method for determining the __values__ of the Q-function for a given __policy__ by defining a direct relationship between the value of __QF[s0,a0]__ and the value of __QF[s1,a1]__.
  
    1. Qn[s,a] = ADD(Prob[s,a,s'] * (r[s,a,s'] + y*Qn[s',a']))
  
  __Dynamic Programming:__ Defines an iterative method for calculating the __Q-Function__, using the __Bellman Equation__. It needs the __State Transition Model (STM)__ of the system (what actions in each state can the agent make & their probabilities if non-deterministic).
  
    1. Bellman Optimality Equation & Principle of Optimality:  The optimal value of:
      •	 Q[s,a] = ADD(Prob[s,a,s'] * (r[s,a,s'] + y*max(Qn[s',a'])))
      
    2. Optimal policy has the property that whatever the initial state and initial action are, the remaining actions must constitute an optimal policy with regards to the state resulting from the first action.
    
    3. Value Iteration Algorithm:

```Python
In:  STM, R, y, Q=0
     while (Qnew – Qold) != 0:
         for each Q[s,a]:
             Q[s,a] = ADD(Prob[s,a,s'] * (r[s,a,s'] + y*max(Qn[s',a'])))
```

__Q-Learning__ defines an approach to finding the optimal Q-Function values without having a prior STM. It learns the Q-Function (rather than calculating it as Dynamic Programming does) on the basis of observed states only and the corresponding received reward:

    1. Q[s,a]' = Q[s,a] + lr * (r[s,a,s'] + y*max(Q[s',a']) - Q[s,a])

It is proved that the learning converges with ***k -> inf***, if all ___[s,a]___ have been ___inf___ often chosen. In real life though - QL can often get stuck depending on the reward topology and learning configuration.

  __Exploration & Exploitation:__ Is a method which deals with trying to converge the QL.
  
    1. Exploitation is when the agent picks the action, which provides it with the max return (max Q-value). Exploitation is needed so the agent uses the QF it has already learnt in order to extract optimal policy and max reward. The problem is the current QF is based on the agent's observations. If they are not close to the "truth", it will get stuck into a "local max" of the reward or never accomplishing its given goal.
    2. Exploration is when the agent picks on random action other than the optimal one. Exploration is needed so the agent can learn the QF distribution and hence be able to find an optimal policy which reflects the real truth.
    3. Epsilon-Greedy Exploration: is an approach to balance exploration & exploitation:

```Python
a = argmax(Q[s,a])
    # Probability = 1 – epsilon (e)
a is random_choice(a != argmax(Q[s,a]))
    # Probability = epsilon (e)
```
  __Q-Learning Algorithm (e-Greedy Exploration):__

```Python
In:  e, R, lr, y, Q=0, s0
     while convergence or trials>N:
         while success or lr<m:
             a = e-Greedy_Exploration()
             s', R[s,a], success = env()
             Q[s,a]' += lr * (R[s,a] + y*max(Q[s',a']) - Q[s,a])
             s=s'
```
                                           
## Experimental Setup
Q-Learning testing is performed for each of the learning configurations: parameter decay functions – (epsilon_fn, LR_fn) ∈ {k_decay, const_k_decay, log_k_decay, const_log_k_decay}; reward discount factor – R_discount ∈ {0.5, 0.9}; with reward=task1.mat. Q-Learning evaluation is performed with (epsilon_fn, LR_fn) = const_k_decay; R_discount = 0.9; with reward=qeval.mat. Additionally, Dynamic Programming is utilized for comparison purposes.

    1. Q-Learn & Test | all configurations, each ran for 10 times.
    2. Dynamic Programming & Test | utilizing reward=task1.mat, auto-generated State-Transition Model.
    3. Q-Learn & Evaluation | the above-mentioned evaluation configuration.
    4. Dynamic Programming & Evaluation | utilizing reward=qeval.mat, auto-generated State-Transition Model.

## Resuts
__For full results & obervations please read [QLearningDynamicProgrammingDoc Report](https://github.com/SamyuelDanyo/QLearningDynamicProgramming/blob/master/docs/QLearningDynamicProgrammingDoc.pdf)__

__For all figures please check [QLearningDynamicProgramming Resources](https://github.com/SamyuelDanyo/QLearningDynamicProgramming/blob/master/res)__


### Q-Learning

__Q-Learning Training__
![Q-Learning Training](/res/ql_pol_train1.PNG)
![Q-Learning Training](/res/ql_pol_train2.PNG)
![Q-Learning Training](/res/ql_pol_train3.PNG)
![Q-Learning Training](/res/ql_pol_train4.PNG)
![Q-Learning Training](/res/ql_pol_train5.PNG)
![Q-Learning Training](/res/ql_pol_train6.PNG)
![Q-Learning Training](/res/ql_pol_train7.PNG)
![Q-Learning Training](/res/ql_pol_train8.PNG)
![Q-Learning Training](/res/ql_pol_train9.PNG)
![Q-Learning Training](/res/ql_pol_train10.PNG)
![Q-Learning Training](/res/ql_pol_train11.PNG)

__Q-Learning Performance__
![Q-Learning Performance](/res/ql_perf.PNG)
![Q-Learning Exploration/Explotation Performance](/res/QL_explr_explt_success_results.PNG)
![Q-Learning Experiment Performance](/res/QL_performance_results.PNG)

__Dynamic Programming Performance__
![Dynamic Programming Performance](/res/dp_perf.PNG)

__Optimal Greedy Policy__
![Optimal Greedy Policy](/res/opt_greedy_pol.PNG)
