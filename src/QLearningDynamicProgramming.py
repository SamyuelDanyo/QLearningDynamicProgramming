#################################################################
# QLearningDynamicProgramming
# Reinforcement Q-Learning for World Grid Navigatio
# Evaluation - use QLearningDynamicProgrammingModule functionality
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
from QLearningDynamicProgrammingModule import *

# EXPERIMENT | EVALUATION
print("Would You Like to Perform Evaluation? (y/n)")
USER_I = input()
if(USER_I in ('yes', 'Yes', 'Y', 'y')):
    # Set the Dataset Args
    filename, reward_data = 'task1.mat', 'reward'
    print("Please Indicate if The Evaluation Args Are Correct? (y/n)")
    print("Filename:{} | Reward Data:{}".format(filename, reward_data))
    USER_I = input()
    if(USER_I not in ('yes', 'Yes', 'Y', 'y')):
        print("Please Enter Filename")
        filename = input()
        print("Please Enter Reward Data Variable Name")
        reward_data = input()
        
    # Get the Reward into NumPy arrays.
    REWARD_DICT = scipy.io.loadmat(filename)
    REWARD = np.array(REWARD_DICT[reward_data])
    print(">>>>>> REWARD MATRIX LOADED <<<<<<")
    
    # Perform Q-Learning
    (QL, opt_policy, opt_policy_coord, reward, greedy_policy,
     goal_reached, run_time, success_rate,
     explr_explt_rate) = perform_QL(state_grid_dim=(10,10), goal_state=(9,9), R_table=REWARD,
                                     idle_state=(0, 0), epsilon_fn=const_k_decay, LR_fn=const_k_decay,
                                     R_discount=0.9, mode='qlearning', ST_TR_M=np.empty(0),
                                     action_trans_conf='standard', max_nb_trials=3000, verbose='medium')
    qevalstates = (opt_policy_coord[:,0]+1) + (opt_policy_coord[:,1]*10)
    
    print("\n>>> QEVALSTATES <<<")
    print(qevalstates)
    print()
	
    # Perform Dynamic Programming
    print("Would You Like to Perform Dynamic Programming? (y/n)")
    USER_I = input()
    if(USER_I in ('yes', 'Yes', 'Y', 'y')):
        (QL, opt_policy, opt_policy_coord,
         reward, greedy_policy, goal_reached,
         run_time) = perform_QL(state_grid_dim=(10,10), goal_state=(9,9), R_table=REWARD,
                                idle_state=(0, 0), epsilon_fn=None, LR_fn=None,
                                R_discount=0.9, mode='dynamic_programming', ST_TR_M=np.empty(0),
                                action_trans_conf='standard', max_nb_trials=3000, verbose='high')