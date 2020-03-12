#################################################################
# QLearningDynamicProgramming
# Reinforcement Q-Learning for World Grid Navigatio
# QLearningDynamicProgramming Module implementing functionality.
# Import in your program and use as a package.
## Q-Learning + Dynamic Programming (Full Implementation)
## QL Helper Functions (Full Implementation)
## Author: Samyuel Danyo
## Date: 18/4/2019
## License: MIT License
## Copyright (c) 2020 Samyuel Danyo
#########################################################
# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
import time
# Set the seed of the numpy random number generator
np.random.seed(seed=1)
import scipy.io
import warnings
import os, sys, operator

# Q-Learning Parameter Control Functions
def k_decay(par=1, k=1):
    return 1.0 / k

def const_k_decay(par=1, k=1, const=100):
    return const / (const + k)

def log_k_decay(par=1, k=1):
    return (1.0 + np.log(k)) / k

def const_log_k_decay(par=1, k=1, const=5):
    return (1.0 + const*np.log(k)) / k

def const(par=1, k=1, const=0.99):
    return const

def const_decay(epsilon=1, k=1, decay_rate=0.99):
    return epsilon*decay_rate

# Environment Class
class Environment:
    """ Environment Class for the Q-Learning Class. """
    def __init__(self, state_grid_dim=(10,10), goal_state=(9,9), R_table=np.empty(0),
                 idle_state=(0, 0), action_trans_conf='standard'):
        """ Args:
                state_grid_dim    (Tuple): Dimensions of the World Grid
                goal_state        (Tuple): Coordinates=(row,col) of the goal state
                R_table           (Numpy Matrix): Reward table
                idle_state        (Tuple): Coordinates=(row,col) of the starting state
                action_trans_conf (String): Action transition configuration:
                    standard - Illegal actions will be assed based on world boundaries & obstacles.
                    reward   - Illegal actions will be assed based on the reward table
                               (-1) rewards will be trated as illegal actions. """
        # Define state space (the gird world)
        self.state_grid_dim = state_grid_dim
        self.goal_state = goal_state
        self.idle_state = idle_state
        self.state = idle_state
        self.obstacle_state = None
       
        # Define action space
        self.action_dim = (4,)
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_trans_conf = action_trans_conf
        
        # Define reward table
        if R_table.size != 0:
            if R_table.shape != (state_grid_dim + self.action_dim):
                self.R = R_table.reshape(self.state_grid_dim + self.action_dim).transpose(1,0,2)
            else: self.R = R_table
        else:
            print("\nNo Reward Table Provided!")
            self.R = self._build_reward_table()
        
        # Check action space consistency
        if len(self.action_dict.keys()) != len(self.action_coords):
            raise ValueError("Inconsistent Action Space!")

    def reset(self):
        """ Reset the agent state. """
        self.state = self.idle_state  
        return self.state

    def step(self, action):
        """ Calculate next state, reward and
            success based on the agent's action. """
        # Compute the next agent location (state)
        state_next = (self.state[0] + self.action_coords[action][0],
                      self.state[1] + self.action_coords[action][1])
        
        # Collect reward
        reward = self.R[self.state + (action,)]
        
        # Terminate if we reach the goal state
        success = (np.array(state_next) == (np.array(self.goal_state))).all()
        
        # Update the agent location (state)
        self.state = state_next
        return state_next, reward, success
    
    def allowed_actions(self):
        """ Generate list of actions allowed
            depending on agent's location. """
        actions_allowed = []
        (y, x) = self.state
        
        if self.action_trans_conf is 'standard':
            # Restricting top boundary
            if (y > 0):
                actions_allowed.append(self.action_dict["up"])
            # Restricting bottom boundary
            if (y < self.state_grid_dim[1] - 1):
                actions_allowed.append(self.action_dict["down"])
            # Restricting left boundary
            if (x > 0):
                actions_allowed.append(self.action_dict["left"])
            # Restricting right boundary
            if (x < self.state_grid_dim[0] - 1):
                actions_allowed.append(self.action_dict["right"])
        
        elif self.action_trans_conf is 'reward':
            # Restricting obstacle from top
            if self.R[y,x,self.action_dict["up"]] != -1:
                actions_allowed.append(self.action_dict["up"])
            # Restricting obstacle from down
            if self.R[y,x,self.action_dict["down"]] != -1:
                actions_allowed.append(self.action_dict["down"])
            # Restricting obstacle from left
            if self.R[y,x,self.action_dict["left"]] != -1:
                actions_allowed.append(self.action_dict["left"])
            # Restricting obstacle from right
            if self.R[y,x,self.action_dict["right"]] != -1:
                actions_allowed.append(self.action_dict["right"])
        
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed

    def _build_reward_table(self):
        """ Define agent reward table R[s,a]. """
        print("Building Reward Table Based on Goal State With 1 Obstacle...")
        # Reward for arriving at goal state
        r_goal = 100
        # Penalty for reaching non-goal state
        r_nongoal = -0.1
        
        # Reward table: R[s,a]
        R_table = r_nongoal * np.ones(self.state_grid_dim + self.action_dim, dtype=float)
        
        # Goal state rewards
        # Enter goal state from top
        if self.goal_state[0] > 0:
            R_table[self.goal_state[0] - 1, self.goal_state[1], self.action_dict["down"]] = r_goal
        # Enter goal state from left
        if self.goal_state[1] > 0:
            R_table[self.goal_state[0], self.goal_state[1] - 1, self.action_dict["right"]] = r_goal
        # Enter goal state from right
        if self.goal_state[1] < self.state_grid_dim[1]-1:
            R_table[self.goal_state[0], self.goal_state[1]+1, self.action_dict["left"]] = r_goal
        # Enter goal state from down
        if self.goal_state[0] < self.state_grid_dim[0]-1:
            R_table[self.goal_state[0]+1, self.goal_state[1], self.action_dict["up"]] = r_goal
        
        # Get a random obstacle coordinates
        obstacle_row = np.random.choice(np.setdiff1d(np.arange(self.state_grid_dim[0]), self.goal_state[0]-1))
        obstacle_col = np.random.choice(np.setdiff1d(np.arange(self.state_grid_dim[1]), self.goal_state[1]-1))
        self.obstacle_state = (obstacle_row, obstacle_col)
        
        # Obstacle rewards
        # Encounter obstacle state from top
        if obstacle_row > 0:
            R_table[obstacle_row - 1, obstacle_col, self.action_dict["down"]] = -1
        # Encounter obstacle state from left
        if obstacle_col > 0:
            R_table[obstacle_row, obstacle_col - 1, self.action_dict["right"]] = -1
        # Encounter obstacle state from right
        if obstacle_col < self.state_grid_dim[1]-1:
            R_table[obstacle_row, obstacle_col+1, self.action_dict["left"]] = -1
        # Encounter obstacle state from down
        if obstacle_row < self.state_grid_dim[0]-1:
            R_table[obstacle_row+1, obstacle_col, self.action_dict["up"]] = -1
        
        return R_table

# Agent Class
class Agent:
    """ Agent Class for the Q-Learning Class. """
    def __init__(self):
        pass
        
    def get_action(self, env, Q, epsilon, explore_count, exploit_count):
        """ Aply Epsilon-greedy agent policy to make an action.
            Args:
                env (Environment class): The Q-Learning environment
                Q   (Numpy Matrix): The Q-function[s,a] of the algorithm.
                epsilon (Float): The current exploration probability
                explore_count (Integer): Number of explorations
                exploit_count (Integer): Number of exploitations """ 
        if np.random.uniform(0, 1) < epsilon:
            # Perform exploration
            explore_count +=1
            state = env.state
            # Get the allowed actions
            actions_allowed = env.allowed_actions()
            # Get the Q-values for the current state
            Q_s = Q[state[0], state[1], actions_allowed]
            # Pick from allowed actions without the highest Q-value
            actions_explore = actions_allowed[np.flatnonzero(Q_s != np.max(Q_s))]
            if actions_explore.size != 0:
                # If all Q-values are the same
                return (np.random.choice(actions_explore), explore_count, exploit_count)
            return (np.random.choice(actions_allowed), explore_count, exploit_count)
        else:
            # Perform exploitation
            exploit_count +=1
            state = env.state
             # Get the allowed actions
            actions_allowed = env.allowed_actions()
             # Get the Q-values for the current state
            Q_s = Q[state[0], state[1], actions_allowed]
            # Pick from allowed actions with the highest Q-value
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return (np.random.choice(actions_greedy), explore_count, exploit_count)
			
# Q-Learning Class
class QLearning():
    """ Q-Learning is unsupervised Machine Learning aproach from the family of Reinforcement Learning algorithms.
        It deals with uncertain and usually unknown system by utilising trial and error
        though exploration and incetive in the form of reward.
        The objective is for the Agent to maximise the received reward while performing a task.
        Q-Learning is based on a few ideas:
        
            Markov Decision Process (MDP): is a model for a dynamic system which exhibit the
                                           so-called Markov property [given the present,
                                           the future does not depend on the past.]
                MDP comprises of a few elements:
                   - Set of states:  S={s1; s2: s}
                   - Set of actions: A={a1; a2: a}
                   - (State) Transition Function: 
                        Deterministic - f : S x A => S
                        Non-deterministic (Probability of the system reaching the new state s'
                                           after taking action a at state s) - f : S x A x S => [0:1]
                   - Reward & Return: When the agent takes an action "a" to make a transition
                                      from state "s" to next state s', reward "r" is assigned by the environment.
                                      The agent receives the reward in the next state s'.
                        - Reward Function: S x A x S => R
                        - Total Reward (Return): The total reward expected in the future by doing k-steps (transitions),
                                                 with future rewards being discounted by a factor of "y".
                            - R0 = r1 + y*r2 + y^2*r3 ... + y^(k-1)*rk
                                           
                            
            Controlling MDP by imposing a decision-making mechanism - Policy:
                Deterministic (Based on a state, an action is taken)
                    - n : S => A
                Non-deterministic (Based on a state and action a probability is given)
                    - n : S x A => [0:1]
             
             Q-Function: Measures the “worth” of taking a specific action "a" at a particular
                         state "s" under a given policy "n" based on the expected Return.
                Qn : S x A => R
            
            Optimal Policy: Maximizes (with respect to a given task) the Q-function over all possible (s,a) pairs.
            
            Bellman Equation: Provides an iterative method for determining the values of the Q-function
                              for a given policy by defining a direct relationship between the value
                              of the Q-function[s0,a0] and the value of the Q-function[s1,a1]
                Qn[s,a] = ADD(Prob[s,a,s'] * (r[s,a,s'] + y*Qn[s',a']))
                
            Dynamic Programming: Defines an iterative method for calculating the Q-Function,
                                 using the Bellman Equation. It needs the State Transition Model (STM)
                                 of the system (what actions in each state can the agent make
                                 & their probabilities if non-deterministic).
                Bellman Optimality Equation & Principle Of Optimality:
                    - The optimal value of Q[s,a] = ADD(Prob[s,a,s'] * (r[s,a,s'] + y*max(Qn[s',a'])))
                    - Optimal policy has the property that whatever the initial state and initial action are,
                      the remaining actions must constitute an optimal policy with regard to the
                      state resulting from the first action.
                    - Value Iteration Alg: In: STM, R, y, Q=0
                                               while Qnew - Qold = 0:
                                                   for each Q[s,a]:
                                                       Q[s,a] = ADD(Prob[s,a,s'] * (r[s,a,s'] + y*max(Qn[s',a'])))
                                                       
        Q-Learning defines an aproach to finding the optimal Q-Function values without having a prior STM.
        It learns the Q-Function (rather than calculating it as Dynamic Programming does) on the basis
        of only observed states and the received reward:
            -  Q[s,a]' = Q[s,a] + lr * (r[s,a,s'] + y*max(Q[s',a']) - Q[s,a])
        It is be proved that the learning converges with k->inf, if all [s,a] have been inf often chosen.
            In real life though - Q-Learning can often get stuck based on the reward topology.
        Exploration & Exploitation: Is a method which deals with trying to converge the Q-Learning.
            - Exploitation is when the agent picks the action, which provides it with the max return (max Q-value).
                  Exploitation is needed so the agent uses the Q-Function it has already learnt in order to
                  extract optimal policy and max reward. The problem is the current Q-Function is based on the
                  agent's observations. If they are not close to the "truth", it will get stuck into a "local max"
                  of the reward or never accomplishing its given goal.
            - Exploration is when the agent picks on random action other than the optimal one.
                  Exploration is needed so the agent can learn the distribution of rewards and hence  be able 
                  to find an optimal policy which reflects the real truth.
        Epsilon-Greedy Exploration: is an aproach to balance exploration & exploitation:
                 a = argmax(Q[s,a])   Probability = 1-epsilon
            a =
                random_choice(a != argmax(Q[s,a])) Probability = epsilon
                
        - Q-Learning Alg (e-Greedy Exploration): In: e, R, lr, y, Q=0, s0
                                                 while convergence or trials>N:
                                                     while success or lr<m:
                                                         a = e-Greedy Exploration()
                                                         s', R[s,a], success = env()
                                                         Q[s,a]' += lr * (R[s,a] + y*max(Q[s',a']) - Q[s,a])
                                                         s=s' 
        """
    def __init__(self, state_grid_dim=(10,10), goal_state=(9,9), R_table=np.empty(0),
                 idle_state=(0, 0), epsilon_fn=const_decay, LR_fn=const, R_discount=0.99,
                 mode='qlearning', ST_TR_M=np.empty(0), action_trans_conf='standard'):
        """ Args:
                state_grid_dim    (Tuple): Dimensions of the World Grid
                goal_state        (Tuple): Coordinates=(row,col) of the goal state
                R_table           (Numpy Matrix): Reward table
                idle_state        (Tuple): Coordinates=(row,col) of the starting state
                epsilon_fn        (Function): Function to define epsilon
                LR_fn             (Function): Function to define learnign rate
                R_discount        (Float): Reward discount factor
                mode              (String): Mode of the algorithm
                    qlearning           - Performs Q-Learning
                    dynamic_programming - Performs Dynamic Programming (STM is needed/generated)
                ST_TR_M           (NumPy Matrix): State Transition Model 
                action_trans_conf (String): Action transition configuration:
                    standard - Illegal actions will be assed based on world boundaries & obstacles.
                    reward   - Illegal actions will be assed based on the reward table
                               (-1) rewards will be trated as illegal actions.
        """
        #################################
        # Environment
        #################################
        self.env = Environment(state_grid_dim, goal_state, R_table, idle_state, action_trans_conf) 
        #################################
        # Agent
        #################################
        self.agent = Agent()
        #################################
        # Mode
        #################################
        self.mode = mode
        if mode is 'dynamic_programming':
            # Define State Transition Model
            if ST_TR_M.size != 0:
                if ST_TR_M.shape != (self.env.state_grid_dim+self.env.action_dim):
                    self.ST_TR_M = ST_TR_M.reshape(self.env.state_grid_dim + self.env.action_dim).transpose(1,0,2)
                else: self.ST_TR_M = ST_TR_M
            else:
                print("\nMode is Dynamic Programming But No State Transition Model Provided!")
                self.ST_TR_M = self._build_state_transition_model()
        #################################
        # Learning configuration
        #################################
        if mode is 'qlearning':
            # Exploration probability
            self.epsilon_fn = epsilon_fn
            self.epsilon = self.epsilon_fn()
            # Learning Rate
            self.LR_fn = LR_fn
            self.LR = LR_fn()
        # Reward discount factor
        self.R_discount = R_discount
        # Initialize Q-function table: Q[s,a]
        self.Q = np.zeros(self.env.state_grid_dim + self.env.action_dim, dtype=float)
    
    def update(self, memory):
        """ Update the Q-function map utilizing the Bellman Equation:
                Q[s,a] <- Q[s,a] + LR * (R[s,a] + R_discount * max(Q[s,:]) - Q[s,a])
            Args:
                memory (Tuple):
                    state      (Tuple): Current agent state (location)
                    action     (Integer): Agent action
                    state_next (Tuple): Next agent state
                    reward     (Float): Current reward
        """
        # Extract the memory
        (state, action, state_next, reward) = memory
        sa = state + (action,)
        # Update the Q-Function
        if self.mode is "qlearning":
            self.Q[sa] += self.LR * (reward + self.R_discount*np.max(self.Q[state_next]) - self.Q[sa])
        elif self.mode is "dynamic_programming":
            if self.ST_TR_M[sa] == 0:
                # Illegal Transition
                self.Q[sa] = 0
            else:
                self.Q[sa] = self.ST_TR_M[sa] * (reward + self.R_discount*np.max(self.Q[state_next]))
        
    def train(self, verbose='low'):
        """ Perform Q-Learning, implementing the Q-Learning algorithm. """
        # Initialisation
        k, reward_full = 1, 0.0
        explore_count, exploit_count = 0, 0
        state = self.env.reset()
        if verbose is 'high':
            policy= []
            policy_coord = []
        # The agent takes action until convergence
        while True:
            # Adjust learning parameters
            self.epsilon = self.epsilon_fn(self.epsilon, k=k)
            self.LR = self.LR_fn(self.LR, k=k)
            # Get action
            (action, explore_count, exploit_count) = self.agent.get_action(
                self.env,self.Q, self.epsilon, explore_count, exploit_count)
            if verbose is 'high': policy.append(action)
            # Compute new state based on the action
            state_next, reward, success = self.env.step(action)
            if verbose is 'high': policy_coord.append(state)
            # Update
            self.update(memory=(state, action, state_next, reward))
            # Next state transition
            state = state_next
            # Feedback data gathering
            k += 1
            reward_full += reward 
            # Convergence check
            if success or self.LR < 0.005:
                break
        if verbose is 'high':
            # Get greedy policy: argmax[a'] Q[s,a']
            greedy_policy = np.zeros(self.env.state_grid_dim, dtype=int)
            for x in range(self.env.state_grid_dim[0]):
                for y in range(self.env.state_grid_dim[1]):
                    greedy_policy[y, x] = np.argmax(self.Q[y, x, :])
            return (k, reward_full, success, explore_count, exploit_count,
                    np.array(policy), np.array(policy_coord), greedy_policy)
        return (k, reward_full, success, explore_count, exploit_count)
    
    def dynamic_programming(self, verbose):
        """ Peroform Dynamic programming, implementing the Value Iteration algorithm.
        """
        print("\nStarting Dynamic Programming...")
        start_time = time.time()
        
        iter_idx = 0
        # Update the Q-Function until convergence
        while True:
            Q_old = np.copy(self.Q)
            # For each Q[s,a]
            for idx_r, row in enumerate(self.Q):
                for idx_c, col in enumerate(row):
                    for idx_a, act in enumerate(col):
                        # Define state & action
                        state = (idx_r, idx_c)
                        action = idx_a
                        state_next = (state[0] + self.env.action_coords[action][0],
                                      state[1] + self.env.action_coords[action][1])
                        # Collect reward
                        reward = self.env.R[state + (action,)]
                        # Update 
                        self.update(memory=(state, action, state_next, reward))
            # Feedback data gathering
            Q_upd = np.sum(np.abs(self.Q - Q_old))
            # Print feedback
            if (verbose in ['medium', 'high']) and (iter_idx % 3 == 0): 
                print("Iter[{}]: Q[sa] Update = {:.2f}"
                  .format(iter_idx+1, Q_upd))
            iter_idx += 1
            # Convergence check
            if -0.1 < Q_upd < 0.01:
                break
            
        print("\nQ Convergence Achieved!\n")
        run_time = time.time() - start_time
        return (iter_idx, run_time)
    
    def _build_state_transition_model(self):
        """ Build simple deterministic state transition model.
            Going out of boundaries is forbidden.
            Going out of goal state is forbidden.
        """
        print("Building Deterministic State Transition Model...")
        # Define topology
        ST_TR_M = np.ones(self.env.state_grid_dim + self.env.action_dim, dtype=float)
        
        # For each (s,a)
        for idx_r, row in enumerate(ST_TR_M):
            for idx_c, col in enumerate(row):
                for idx_a, act in enumerate(col):
                    state = (idx_r,idx_c)
                    n_state = np.array((idx_r,idx_c)) + np.array(self.env.action_coords[idx_a])
                    # Ban moves out of the grid world
                    if np.any(n_state < 0) or np.any(n_state == np.array(self.env.state_grid_dim)):
                        ST_TR_M[idx_r, idx_c, idx_a] = 0
                    # Ban moves out of the goal state
                    if state == self.env.goal_state:
                        ST_TR_M[idx_r, idx_c] = 0
                    # Ban moves into an obstacles
                    if self.env.obstacle_state:
                        if np.all(n_state == self.env.obstacle_state):
                            ST_TR_M[idx_r, idx_c, idx_a] = 0
        return ST_TR_M
        
    def get_optimal_policy(self):
        """ Get Optimal Policy:
            Returns:
                greedy_policy    (NumPy Matrix): for each s: a = argmax(Q[s,a'])
                opt_policy       (NumPy Array): Optimal action path from idle to goal state
                opt_policy_coord (NumPy Matrix): Optimal state-coord path from idle to goal state """
        # Initialise greedy policy topology
        greedy_policy = np.zeros(self.env.state_grid_dim, dtype=int)
        # Calculate the greedy policy
        for x in range(self.env.state_grid_dim[0]):
            for y in range(self.env.state_grid_dim[1]):
                greedy_policy[y, x] = np.argmax(self.Q[y, x, :])
        
        # Initialisation Optimal Policy
        state = self.env.idle_state
        reward = 0.0
        goal_reached = False
        opt_policy= []
        opt_policy_coord = []
        while True:
            # Get action
            action = greedy_policy[state]
            # Ssave the policy
            opt_policy.append(action)
            opt_policy_coord.append(np.array(state))
            # Get next stae
            state_next = (state[0] + self.env.action_coords[action][0],
                          state[1] + self.env.action_coords[action][1])
            # Collect reward
            reward += self.env.R[state + (action,)]
        
            # Terminate if we reach the goal state
            if (np.array(state_next) == (np.array(self.env.goal_state))).all():
                goal_reached = True
                opt_policy_coord.append(np.array(self.env.goal_state))
                break
            # Terminate if we get stuck
            elif len(opt_policy)>self.env.state_grid_dim[0]*self.env.state_grid_dim[1]:
                break
        
            # Update the agent location (state)
            state = state_next
            
        return (np.array(opt_policy), np.array(opt_policy_coord),
                reward, greedy_policy, goal_reached)
    
    def display_policy(self, policy, policy_coord,
                       reward, greedy_policy):
        """ Plot a policy & display greedy policy.
        """
        print("\nGreedy Policy (y, x):")
        print(greedy_policy)
        print()
        for (key, val) in sorted(self.env.action_dict.items(), key=operator.itemgetter(1)):
            print(" action['{}'] = {}".format(key, val))
        # Initialisation
        plot_action_dict = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        (X, Y) = self.env.state_grid_dim
        # Plot the grid world
        plt.figure(figsize=(X,Y))
        for idx, coord in enumerate(policy_coord):
            if idx == (len(policy_coord)-1):
                break
            # Plot policy
            plt.text(coord[1]+.3, X - 1 - coord[0]+.35,  plot_action_dict[policy[idx]],
                      color='r', fontdict={'weight': 'bold',  'size': 25})
        # Indicate goal state
        plt.text(self.env.goal_state[1]+.25, X - 1 - self.env.goal_state[0]+.15,  '★',
                      color='y', fontdict={'weight': 'bold',  'size': 34})
        # Indicate idle state (start)
        plt.text(self.env.idle_state[1]+0.1, X - 1 - self.env.idle_state[0]+0.7,  'start',
                      color='b', fontdict={'weight': 'bold',  'size': 17})
        # Indicate obstacle state
        if self.env.obstacle_state:
            plt.text(self.env.obstacle_state[1]+.25, X - 1 - self.env.obstacle_state[0]+.35,  '■',
                      color='black', fontdict={'weight': 'bold',  'size': 35})
        plt.axis([0, X, 0,  Y])
        plt.xticks(range(X+1))
        plt.yticks(range(Y+1))
        plt.xlabel('ROW', fontsize=13)
        plt.ylabel('COL', fontsize=13)
        plt.suptitle('Q-Learning Policy [Reward={}, k={}]'.format(reward, len(policy)), size=16, y=0.93)
        plt.legend(loc='best')
        plt.grid()
        plt.show()
		
# Q-Learning Helper Functions
def perform_QL(state_grid_dim=(10,10), goal_state=(9,9), R_table=np.empty(0),
               idle_state=(0, 0), epsilon_fn=const_decay, LR_fn=const,
               R_discount=0.99, mode='qlearning', ST_TR_M=np.empty(0),
               action_trans_conf='standard', max_nb_trials=500, verbose='low'):
    """ Perform Q-Learning.
            Args:
                state_grid_dim    (Tuple): Dimensions of the World Grid
                goal_state        (Tuple): Coordinates=(row,col) of the goal state
                R_table           (Numpy Matrix): Reward table
                idle_state        (Tuple): Coordinates=(row,col) of the starting state
                epsilon_fn        (Function): Function to define epsilon
                LR_fn             (Function): Function to define learnign rate
                R_discount        (Float): Reward discount factor
                mode              (String): Mode of the algorithm
                    qlearning           - Performs Q-Learning
                    dynamic_programming - Performs Dynamic Programming (STM is needed/generated)
                ST_TR_M           (NumPy Matrix): State Transition Model 
                action_trans_conf (String): Action transition configuration:
                    standard - Illegal actions will be assed based on world boundaries & obstacles.
                    reward   - Illegal actions will be assed based on the reward table
                               (-1) rewards will be trated as illegal actions.
                max_nb_trials     (Integer): Maximum number of trials
                verbose           (String): Verbosity level {false, low, medium, high} """
    # Initialise the Q-Learning object
    QL = QLearning(state_grid_dim, goal_state, R_table,
                   idle_state, epsilon_fn, LR_fn, R_discount, mode, ST_TR_M, action_trans_conf)
    
    # Perform Q-Learning
    if QL.mode is 'qlearning':
        print("Training...\n")
        start_time = time.time()
        
        # Initialisation of feedback data gathering
        success_nb, max_reward = 0, 0.0
        explore_count_t, exploit_count_t = 0, 0
        rewards = []
        
        # Learning until convergence or max number of trials
        for trial in range(max_nb_trials):
            # Train for the trial
            if verbose is 'high':
                (k, reward, success, explore_count, exploit_count,
                 policy, policy_coord, greedy_policy) = QL.train(verbose)
            else:
                (k, reward, success, explore_count, exploit_count) = QL.train(verbose)
            # Data gathering
            success_nb += success
            rewards.append(reward)
            explore_count_t += explore_count
            exploit_count_t += exploit_count
            if reward > max_reward:
                max_reward = reward

            # Print training feedback
            if (verbose in ['medium', 'high']) and (trial % int(max_nb_trials/20) == 0): 
                print("Trial[{}/{}]: Success = {}, k = {}, Reward = {:.1f}"
                      .format(trial + 1, max_nb_trials, success, k, reward))
                print("                  Explore/Exploit[{}/{}]".format(explore_count, exploit_count))
                if (verbose is 'high') and trial > 5:
                    QL.display_policy(policy, policy_coord,
                                  reward, greedy_policy)
            # Check convergence
            if ((len(rewards) > 3 and (rewards[-1] == rewards[-2] == rewards[-3] == max_reward))
                or (k <= np.array(state_grid_dim).sum() and success)):
                break

        print("\nLearning Completed!")
        run_time = time.time() - start_time
        
        # Get optimal policy
        (opt_policy, opt_policy_coord, reward, greedy_policy, goal_reached) = QL.get_optimal_policy()
        
        # Pring total learning feedback
        success_rate = success_nb/(trial+1) * 100
        explr_explt_rate = explore_count_t / exploit_count_t * 100
        m, s = divmod(run_time, 60)
        h, m = divmod(m, 60)
        if verbose in ['low', 'medium', 'high']:
            print("\n========================================================================")
            print("Q-LEARNING CONFIGURATION:")
            print("    Epsilon Function = {}".format(QL.epsilon_fn))
            print("    Learning Rate Function = {}".format(QL.LR_fn))
            print("    Reward Discount Factor = {}".format(QL.R_discount))
            print("    State Grid Dims = {}:".format(QL.env.state_grid_dim))
            if QL.env.obstacle_state:
                print("      States: Idle = [{}] | Goal = [{}] | Obstacle = [{}]"
                      .format(QL.env.idle_state, QL.env.goal_state, QL.env.obstacle_state))
            else:
                print("      States: Idle = [{}] | Goal = [{}]".format(QL.env.idle_state, QL.env.goal_state))
            print("LEARNING RESULTS:")
            print("    Goal Reached = {}".format(goal_reached))
            print("    Training Time: {}h:{}m:{:.2f}s | Trials: {}".format(h,m,s,trial+1))
            print("    Success Rate: {}/{} ({:.2f}%) | Max Reward: {:.1f}"
                  .format(success_nb, trial+1, success_rate, max_reward))
            print("    k-Movements: {} | Explore/Exploit Rate[{}/{}] ({:.2f}%)"
                  .format(k, explore_count_t, exploit_count_t, explr_explt_rate))
            print("========================================================================")
            
            if (verbose in ['medium', 'high']) or (verbose is 'low' and goal_reached):
                # Display optimal policy
                QL.display_policy(opt_policy, opt_policy_coord,
                                   reward, greedy_policy)
            
        run_time=np.array([h,m,s])
        return (QL, opt_policy, opt_policy_coord, reward, greedy_policy, goal_reached,
                run_time, success_rate, explr_explt_rate)
    
    # Perform Dynamic Programming
    elif QL.mode is 'dynamic_programming':
        (iter_idx, run_time) = QL.dynamic_programming(verbose)
        
        # Get optimal policy
        (opt_policy, opt_policy_coord, reward, greedy_policy, goal_reached) = QL.get_optimal_policy()
    
        # Pring feedback
        m, s = divmod(run_time, 60)
        h, m = divmod(m, 60)
        if verbose in ['low', 'medium', 'high']:
            print("========================================================================")
            print("DYNAMIC PROGRAMMING CONFIGURATION:")
            print("    State Grid Dims = {}:".format(state_grid_dim))
            if QL.env.obstacle_state:
                print("      States: Idle = [{}] | Goal = [{}] | Obstacle = [{}]"
                      .format(idle_state, goal_state, QL.env.obstacle_state))
            else:
                print("      States: Idle = [{}] | Goal = [{}]".format(idle_state, goal_state))
            print("LEARNING RESULTS:")
            print("    Goal Reached = {}".format(goal_reached))
            print("    Training Time: {}h:{}m:{:.2f}s | Iterations: {}".format(h,m,s,iter_idx))
            print("========================================================================")
            
            if (verbose in ['medium', 'high']) or (verbose is 'low' and goal_reached):
                # Display optimal policy
                QL.display_policy(opt_policy, opt_policy_coord,
                                   reward, greedy_policy)
            
        run_time=np.array([h,m,s])
        return (QL, opt_policy, opt_policy_coord, reward, greedy_policy, goal_reached, run_time)