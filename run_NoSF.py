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

def no_SF(obs):
    #action_space = [create_matrix(3, 0) for _ in range(line_num)]
    action_space = [[] for _ in range(line_num)]
    for n in range(line_num):
        target_product = 0
        target_layer = 0
        target_qty = 0
        action_space[n].append(target_product)
        action_space[n].append(target_layer)
        action_space[n].append(target_qty)
    return action_space

def run_noSF(env):
    SF_decision_ls = []
    production_ls = []
    sum_TH = 0
    sum_ME = 0
    states, obs = env.reset() 
    done = False
    while not done:     
        action_space = no_SF(obs)
        feasible, obs_, states_, rewards, done, step, bl_table, production_qty, backlog, sending_qty, SF_decision, begin_WIP, shift_machine_dedication = env.step(action_space)
        obs = copy.deepcopy(obs_)
        SF_decision_ls.append(SF_decision)
        production_ls.append(production_qty)
        sum_TH +=  sum(production_qty.values())
        sum_ME += return_ME(production_qty, SF_decision, shift_machine_dedication)
    final_end_WIP = begin_WIP
    remain_backlog = return_remain_backlog(final_end_WIP, bl_table)
    avg_TH = sum_TH/period_num
    avg_ME = sum_ME/period_num
    return avg_TH, avg_ME, remain_backlog, bl_table, begin_WIP, SF_decision_ls, production_ls    


start_time = time.time()
env = Fab_env(3, test_data)
avg_TH, avg_ME, remain_backlog, bl_table, begin_WIP, SF_decision, production_ls = run_noSF(env) 
end_time = time.time()
run_time = end_time - start_time

print("Avg_TH: ", avg_TH)
print("Avg_ME: ", avg_ME)
print("Remain_backlog: ", remain_backlog)
print("Avg_CPU_time: ", run_time)





























