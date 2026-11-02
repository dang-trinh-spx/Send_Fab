from Env.fab_env import Fab_env
import numpy as np
import copy
from data_preprocessing import line_num, product_num, layer_num, batch_qty, period_num, machine_num, processing_time
from data_preprocessing import create_3D_matrix, create_matrix
import time
from test_setup import return_ME, return_remain_backlog, test_length, test_data

#Define function to run randomly
def run_random(env):
    SF_decision_ls = []
    sum_TH = 0
    sum_ME = 0
    states, obs = env.reset() 
    done = False
    while not done: 
        action_space = [[] for _ in range(line_num)]
        for n in range(line_num):
            action_space[n] = env.action_space[n].sample()
        feasible, obs_, states_, rewards, done, step, bl_table, production_qty, backlog, sending_qty, SF_decision, begin_WIP, shift_machine_dedication = env.step(action_space)
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

#Evaluate trained policy on test datasets (unseen dataset)
for i in range(test_length):   
        test_avg_TH = []
        test_avg_ME = []
        test_remain_backlog = []
        cpu_time = []
        SF_decision_ls = []
        start_time = time.time()
        env = Fab_env(3, test_data)
        avg_TH, avg_ME, remain_backlog, bl_table, begin_WIP, SF_decision = run_random(env) #<- Uncomment if run random without RL
        test_avg_TH.append(avg_TH)
        test_avg_ME.append(avg_ME)
        test_remain_backlog.append(remain_backlog)
        end_time = time.time()
        run_time = end_time - start_time
        cpu_time.append(run_time)

print("Avg_TH: ", np.mean(test_avg_TH))
print("Avg_ME: ", np.mean(test_avg_ME))
print("Remain_backlog: ", np.mean(test_remain_backlog))
print("Avg_CPU_time: ", np.mean(cpu_time))














