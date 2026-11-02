from Env.fab_env import Fab_env
import numpy as np
import math
import copy
from data_preprocessing import product_num, layer_num, batch_qty, period_num
from spyder_kernels.utils.iofuncs import load_dictionary as ld
import time
import random
from metrics_calculate import return_ME, return_remain_backlog, return_throughput
import argparse
from experiment_design import train_data_ls, eval_data


#Train and Test data list
data_1 = train_data_ls[9]
data_2 = train_data_ls[11]
data_3 = train_data_ls[12]
data_4 = train_data_ls[16]
data_5 = train_data_ls[18]
data_6 = train_data_ls[24]
data_7 = train_data_ls[30]
data_8 = train_data_ls[31]
data_9 = train_data_ls[36]
data_10 = train_data_ls[39]

train_data_ls = [data_1,
                 data_2,
                 data_3,
                 data_4,
                 data_5,
                 data_6,
                 data_7,
                 data_8,
                 data_9,
                 data_10]


#Initiate training model params
parser = argparse.ArgumentParser()
parser.add_argument("--agent_num", default=3, type=int)
parser.add_argument("--train_num", default=5000, type=int)
parser.add_argument("--train_episode", default=20, type=int)
parser.add_argument("--eval_freq", default=20, type=int)
parser.add_argument("--save_freq", default=1000, type=int)
config = parser.parse_args()

#Function to kick-off training
def rollout(agent, train_data_ls, eval_data, saving_dir, config, agent_type):
    
    #Initiate RL agents
    n_actions = np.array([product_num,layer_num,len(batch_qty)], dtype=np.int64)
    agent = agent(n_actions = n_actions, chkpt_dir = saving_dir)
    
    reward_hist = []
    TH_hist = []
    ME_hist = []
    backlog_hist = []
    best_reward = -math.inf
    test_step = 0
    train_step = 0
    
    print("----------Start Training----------")
    
    for i in range(1, config.train_num + 1):
        instance = random.randint(0, 9)
        env = Fab_env(agent_nb = config.agent_num, input_data = train_data_ls[instance])
        state, obs = env.reset() 
        done = False
        start_time = time.time()
        while not done:
            actions, probs, vals = agent.choose_action(state, obs)
            feasible, obs_, state_, rewards, done, step, bl_table, production_qty, backlog, sending_qty, SF_decision, begin_WIP = env.step(actions)
            if feasible != 0:
                agent.remember(state, obs, actions, probs, vals, rewards, done)
                obs = copy.deepcopy(obs_)
                state = copy.deepcopy(state_)
            else:
                done = True
        end_time = time.time()
        run_time = end_time - start_time
        print("Train iters ", i," ", run_time, "Instance: ", instance)
        
        #Update parameters of NNs after a number of episodes
        if i % config.train_episode == 0:    
            total_loss = agent.learn()
            train_step += config.train_episode
            avg_total_loss = (np.array(total_loss)).mean()
            print("training steps: {0}, total_loss: {1}".format(train_step, avg_total_loss))
            
            #Clear memory
            print("... clearing memory ...")
            for a in range(config.agent_num):
                agent.memory[a].clear_memory()
            agent.critic_memory.clear_memory()
         
        #Evaluate trained policy after a number of episodes    
        if i % config.eval_freq == 0:
            test_step += config.eval_freq
            print("---------Evaluation Begin---------")
            test_start_time = time.time()
            avg_TH, avg_ME, remain_backlog, ep_reward = evaluate(agent, eval_data)
            test_end_time = time.time()
            test_run_time = test_end_time - test_start_time

            #Add to draw graph
            reward_hist.append(ep_reward)
            TH_hist.append(avg_TH)
            ME_hist.append(avg_ME)
            backlog_hist.append(remain_backlog)
                        
            print("Test steps: {0}, avg_TH: {1}, avg_ME: {2}, avg_remain_backlog: {3}".format(test_step, avg_TH, avg_ME, remain_backlog))
            print("Run_time: {0}".format(test_run_time)) 
            
            mean_reward = ep_reward
            if mean_reward > best_reward:
                agent.save_models()
                best_reward = mean_reward
            print("mean reward: ", mean_reward, "best reward: ", best_reward)
        
        #Saving
        if i % config.save_freq == 0:
            print("---------Checkpoint---------")
            agent.save_models()
        
    return reward_hist, TH_hist, ME_hist, backlog_hist, train_step


#Define evaluate function to calculate throughput, ME and backlog qty                            
def evaluate(policy, env, agent_type):
    sum_TH = 0
    sum_ME = 0
    ep_reward = 0
    env = Fab_env(agent_nb = config.agent_nb, input_data = env)
    state, obs = env.reset()
    done = False
    while not done:
        actions = policy.evaluate_action(obs) #<- In the evaluation step, agent choose action with highest probs
        feasible, obs_, state_, rewards, done, step, bl_table, production_qty, backlog, sending_qty, SF_decision, begin_WIP = env.step(actions)
        if feasible == 0:
            actions = policy.choose_action(state, obs) 
            feasible, obs_, state_, rewards, done, step, bl_table, production_qty, backlog, sending_qty, SF_decision, begin_WIP = env.step(actions)
            
        obs = copy.deepcopy(obs_)
        state = copy.deepcopy(state_)
        
        #Calculate episode reward
        if agent_type == 'MAPPO':
            ep_reward += rewards[0] #MAPPO rewards
        elif agent_type == 'IPPO':
            ep_reward += sum(rewards) #IPPO rewards
            
        sum_TH +=  return_throughput(production_qty, SF_decision)
        sum_ME += return_ME(production_qty, SF_decision)
        
    final_end_WIP = begin_WIP
    remain_backlog = return_remain_backlog(final_end_WIP, bl_table)
    avg_TH = sum_TH/period_num
    avg_ME = sum_ME/period_num    
    return avg_TH, avg_ME, remain_backlog, ep_reward











