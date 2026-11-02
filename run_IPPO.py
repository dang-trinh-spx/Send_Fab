from Env.fab_env import Fab_env
import numpy as np
from spyder_kernels.utils.iofuncs import load_dictionary as ld
import math
import matplotlib.pyplot as plt
import copy
from data_preprocessing import line_num, product_num, layer_num, batch_qty, period_num, machine_num, processing_time
from data_preprocessing import create_3D_matrix, create_matrix
import argparse
import time
from test_setup import return_ME, return_remain_backlog, test_length, test_data
from RL_Algorithm.ippo import Agent


#Define action number
n_actions = np.array([product_num,layer_num,len(batch_qty)], dtype=np.int64)

#Load trained agents
agents = Agent(n_actions = n_actions, chkpt_dir = 'trained_IPPO/4P/NNs') #<- Replace with your directoty, choose the saved model you want to load
agents.load_models()

#Define function to run RL
def run_RL(env, agents):
    sum_TH = 0
    sum_ME = 0
    actions_ls = []
    sending_qty_ls = []
    SF_decision_ls = []
    probs_ls = []
    vals_ls = []
    states, obs = env.reset() 
    done = False
    total_reward = 0
    while not done: 
        actions = agents.evaluate_action(obs)
        feasible, obs_, states_, rewards, done, step, bl_table, production_qty,\
        backlog, sending_qty, SF_decision, begin_WIP, shift_machine_dedication = env.step(actions)
        obs = copy.deepcopy(obs_)
        states = copy.deepcopy(states_)
        total_reward += sum(rewards)
        actions_ls.append(actions)
        sum_TH +=  sum(production_qty.values())
        if SF_decision:
            sum_TH += sum(SF_decision.values())
        sum_ME += return_ME(production_qty, SF_decision, shift_machine_dedication)
        sending_qty_ls.append(sending_qty)
        SF_decision_ls.append(SF_decision)
    final_WIP = begin_WIP
    remain_backlog = return_remain_backlog(final_WIP, bl_table)
    avg_TH = sum_TH/period_num
    avg_ME = sum_ME/period_num
    return avg_TH, avg_ME, remain_backlog, actions_ls, probs_ls, vals_ls, SF_decision_ls, bl_table

#Evaluate trained policy on test datasets (unseen dataset)
start_time = time.time()
env = Fab_env(3, test_data)
avg_TH, avg_ME, remain_backlog, actions_ls, probs_ls, vals_ls, SF_decision_ls, bl_table = run_RL(env, agents)
end_time = time.time()
run_time = end_time - start_time


print("Avg_TH: ", avg_TH)
print("Avg_ME: ", avg_ME)
print("Remain_backlog: ", remain_backlog)
print("Avg_CPU_time: ", run_time)





















