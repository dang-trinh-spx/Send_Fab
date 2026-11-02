from environment import MD_fab_env, D_fab_env, mod_MILP
import numpy as np
import copy
from data_preprocessing import product_num, layer_num, batch_qty, period_num
import time
from metrics_calculate import return_ME, return_remain_backlog, return_throughput
from experiment_design import test_data
from RL_Algorithm.MD_MAPPO import Agent as MD_MAPPO_agent
from RL_algorithm.D_MAPPO import Agent as D_MAPPO_agent
import pickle
from pathlib import Path


#Load trained agents
agent_num = 3 #<- Change in case the number of agents changes
MD_actions = np.array([product_num,layer_num,len(batch_qty)], dtype=np.int64)

## Multi-discrete MAPPO
agent_1_info = ['Multi-discrete_MAPPO', 'results/MAPPO/MD_MAPPO']
agent_1 = MD_MAPPO_agent(n_actions = MD_actions, chkpt_dir = 'FILE_LOCATION') #<- Replace with your directoty, choose the saved model you want to load
agent_1.load_models()

##Multi-discrete IPPO
agent_2_info = ['Multi-discrete_IPPO', 'results/IPPO/MD_MAPPO']

## Discrete MAPPO
agent_3_info = ['Discrete_MAPPO', 'results/MAPPO/D_MAPPO']

## Discrete IPPO
agent_4_info = ['Discrete_IPPO', 'results/IPPO/D_MAPPO']


#Define function to run RL
def run_RL(env, agents):
    sum_TH = 0
    sum_ME = 0
    actions_ls = []
    sending_qty_ls = []
    SF_decision_ls = []
    production_qty_ls = []
    backlog_ls = []
    probs_ls = []
    vals_ls = []
    states, obs = env.reset() 
    done = False
    total_reward = 0
    while not done: 
        actions, probs, vals = agents.choose_action(states, obs)
        #actions = agents.evaluate_action(obs)        
        feasible, obs_, states_, rewards, done, step, bl_table, production_qty,\
        backlog, sending_qty, SF_decision, begin_WIP, shift_machine_dedication = env.step(actions)
        obs = copy.deepcopy(obs_)
        #states = copy.deepcopy(states_)
        total_reward += sum(rewards)
        actions_ls.append(actions)
        
        #Calculate total TH + ME
        sum_TH +=  return_throughput(production_qty, SF_decision)
        sum_ME += return_ME(production_qty, SF_decision, shift_machine_dedication)
        
        sending_qty_ls.append(sending_qty)
        SF_decision_ls.append(SF_decision)
        production_qty_ls.append(production_qty)
        backlog_ls.append(backlog)
                
    #Update beginning WIP for next shift    
    final_WIP = begin_WIP
    
    #Calculate remaining backlog qty
    remain_backlog = return_remain_backlog(final_WIP, bl_table)
    
    #Calculate average TH + ME per shift
    avg_TH = sum_TH/period_num
    avg_ME = sum_ME/period_num
    return avg_TH, avg_ME, remain_backlog, actions_ls, probs_ls, vals_ls, SF_decision_ls, bl_table, production_qty_ls, backlog_ls

#Evaluate trained policy on test datasets (unseen dataset)
def RL_solving(input_info, input_env, input_agent, input_data):
    start_time = time.time()
    
    #Define environment + agent type
    env = input_env(agent_num, input_data)
    agent = input_agent
    
    #Run RL interacting loop
    avg_TH,\
    avg_ME,\
    remain_backlog,\
    actions_ls,\
    probs_ls,\
    vals_ls,\
    SF_decision_ls,\
    bl_table,\
    production_qty_ls,\
    backlog_ls = run_RL(env, agent)
    
    #CPU time
    end_time = time.time()
    run_time = end_time - start_time
    
    #Export results
    print("Results of ", input_info[0])
    print("Avg_TH: ", avg_TH)
    print("Avg_ME: ", avg_ME)
    print("Remain_backlog: ", remain_backlog)
    print("Avg_CPU_time: ", run_time)
    
    #Define saving directory
    saving_dir = Path(input_info[1])
    file_1 = Path('throughput.pkl')
    file_2 = Path('SendFAB_qty.pkl')
    file_3 = Path('backlog.pkl')
    
    with open(saving_dir / file_1, 'wb') as f:
        pickle.dump(production_qty_ls, f)        
            
    with open(saving_dir / file_2, 'wb') as f:
        pickle.dump(SF_decision_ls, f)    

    with open(saving_dir / file_3, 'wb') as f:
        pickle.dump(backlog_ls, f)  

















