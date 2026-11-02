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

def choose_layer(obs):
    #action_space = [create_matrix(3, 0) for _ in range(line_num)]
    action_space = [[] for _ in range(line_num)]
    for n in range(line_num):
        WIP_level = obs[n][0]
        line_bl = obs[n][1]
        max_val = 0
        target_product = 0
        target_layer = 0
        target_qty = 0
        for i in range(product_num):
            for j in range(layer_num):
                if line_bl[i,j] > max_val: #Max 200
                    max_val = line_bl[i,j]
                    target_product = i
                    target_layer = j
                    for l in range(len(batch_qty)):
                        if batch_qty[l] <= max_val:
                            target_qty = l
                            
        action_space[n].append(target_product)
        action_space[n].append(target_layer)
        action_space[n].append(target_qty)
    return action_space

def run_llq(env):
    SF_decision_ls = []
    sum_TH = 0
    sum_ME = 0
    states, obs = env.reset() 
    done = False
    while not done:     
        action_space = choose_layer(obs)
        feasible, obs_, states_, rewards, done, step, bl_table, production_qty, backlog, sending_qty, SF_decision, begin_WIP, shift_machine_dedication = env.step(action_space)
        obs = copy.deepcopy(obs_)
        states = copy.deepcopy(states_)
        SF_decision_ls.append(SF_decision)
        sum_TH +=  sum(production_qty.values())
        if SF_decision:
            sum_TH += sum(SF_decision.values())
        sum_ME += return_ME(production_qty, SF_decision, shift_machine_dedication)
    final_end_WIP = begin_WIP
    remain_backlog = return_remain_backlog(final_end_WIP, bl_table)
    avg_TH = sum_TH/period_num
    avg_ME = sum_ME/period_num
    return avg_TH, avg_ME, remain_backlog, bl_table, begin_WIP, SF_decision_ls    


start_time = time.time()
env = Fab_env(3, test_data)
avg_TH, avg_ME, remain_backlog, bl_table, begin_WIP, SF_decision = run_llq(env) #<- Uncomment if run LLQ
end_time = time.time()
run_time = end_time - start_time

print("Avg_TH: ", avg_TH)
print("Avg_ME: ", avg_ME)
print("Remain_backlog: ", remain_backlog)
print("Avg_CPU_time: ", run_time)





























