#################################################################
# QLearningDynamicProgramming
# Reinforcement Q-Learning for World Grid Navigation
# Experiment with policy learning and hyper-parameter tunning
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

# Get the Reward into NumPy arrays.
REWARD_DICT = scipy.io.loadmat('task1.mat')
REWARD = np.array(REWARD_DICT["reward"])
print(">>>>>> REWARD MATRIX LOADED <<<<<<")

# EXPERIMENT | TEST
# CONTROL HYPERPARAMETERS 
Par_fn_list = [k_decay, const_k_decay, log_k_decay, const_log_k_decay]
R_discount_list = [0.5, 0.9]

# Q-Learning
# RUN EXPERIMENT
N = 10
teststates = []
greedystates = []
success_rates = []
for par_fn in Par_fn_list:
    for r_discount in R_discount_list:
        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Reward Discount Factor = {}".format(r_discount))
        print("Epsilon/LR Decay Function: {}".format(par_fn))
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        run_success_rate = 0
        success_run_time = np.array([0.0,0.0,0.0])
        av_run_time = np.array([0.0,0.0,0.0])
        run_explr_explt_rate = 0.0
        for run in range(N):
            print("\n>>>>> RUN:{} <<<<<".format(run))
            (QL, opt_policy, opt_policy_coord, reward,
             greedy_policy, goal_reached, run_time,
             success_rate, explr_explt_rate) = perform_QL(state_grid_dim=(10,10), goal_state=(9,9), R_table=REWARD,
                                     idle_state=(0, 0), epsilon_fn=par_fn, LR_fn=par_fn,
                                     R_discount=r_discount, mode='qlearning', ST_TR_M=np.empty(0),
                                     action_trans_conf='standard', max_nb_trials=3000, verbose='False')
            run_explr_explt_rate += explr_explt_rate
            av_run_time += run_time
            
            if goal_reached:
                run_success_rate += goal_reached
                success_run_time += run_time
                if run_success_rate == 1:
                    print("\n>>> OPTIMAL POLICY FOUND <<<")
                    teststates.append([(opt_policy_coord[:,0]+1) * (opt_policy_coord[:,1]+1), reward])
                    greedystates.append(greedy_policy)
                    QL.display_policy(opt_policy, opt_policy_coord,
                                      reward, greedy_policy)
            
            if run == (N-1) and run_success_rate == 0:
                print("\n>>> NO OPTIMAL POLICY FOUND <<<")
                teststates.append([None, reward])
                greedystates.append(greedy_policy)
                
            if run == (N-1):
                run_explr_explt_rate /= N
                av_run_time = av_run_time / N
                print("\nRUN SUCCESS RATE {}/{} ({:.2f}%)"
                      .format(run_success_rate, N, run_success_rate/N*100))
                print("SUCCESSFUL RUN EXECUTION TIME {}h:{}m:{:.2f}s"
                      .format(success_run_time[0], success_run_time[1], success_run_time[2]))
                print("AVERAGE RUN EXECUTION TIME {}h:{}m:{:.2f}s"
                      .format(av_run_time[0], av_run_time[1], av_run_time[2]))
                print("AVERAGE EXPLORE/EXPLOIT RATE {:.2f}%".format(run_explr_explt_rate))
                run_success_rate /= N
                success_rates.append(run_success_rate)
		
# Dynamic Programming
(QL, opt_policy, opt_policy_coord,
 reward, greedy_policy, goal_reached,
 run_time) = perform_QL(state_grid_dim=(10,10), goal_state=(9,9), R_table=REWARD,
                        idle_state=(0, 0), epsilon_fn=None, LR_fn=None,
                        R_discount=0.9, mode='dynamic_programming', ST_TR_M=np.empty(0),
                        action_trans_conf='standard', max_nb_trials=3000, verbose='low')
if goal_reached:
    teststates.append([(opt_policy_coord[:,0]+1) * (opt_policy_coord[:,1]+1), reward])
else: teststates.append([None, reward])
greedystates.append(greedy_policy)
success_rates.append(1.0)
