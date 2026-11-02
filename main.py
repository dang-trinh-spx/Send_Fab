from Env.fab_env import Fab_env
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from data_preprocessing import product_num, layer_num, batch_qty, period_num
import time
import random
import pickle
from RL_frame import rollout, train_data_ls, eval_data, config
from RL_algorithm.MD_MAPPO import Agent as MD_MAPPO_agent


#Define training model (Train 1 model at a time)
train_model = [MD_MAPPO_agent,'trained_NNs/MAPPO/MD_NNs','MAPPO']
#train_model = [MD_IPPO_agent, 'trained_NNs/IPPO/MD_NNs', 'IPPO']
#train_model = [D_MAPPO_agent, 'trained_NNs/MAPPO/D_NNs', 'MAPPO']
#train_model = [D_IPPO_agent, 'trained_NNs/IPPO/D_NNs', 'IPPO']

if __name__ == '__main__':
         reward_hist, TH_hist, ME_hist, backlog_hist, train_step = rollout(
             agent = train_model[0], 
             train_data_ls= train_data_ls, 
             eval_data= eval_data, 
             config = config,
             saving_dir = train_model[1],
             agent_type = train_model[2]) 

#Save results
with open('results/discrete_action/reward.pkl', 'wb') as f: #Sample FILE_LOCATION
    pickle.dump(reward_hist, f)
    
with open('FILE_LOCATION/th.pkl', 'wb') as f:
    pickle.dump(TH_hist, f)

with open('FILE_LOCATION/me.pkl', 'wb') as f:
    pickle.dump(ME_hist, f)

with open('FILE_LOCATION/backlog.pkl', 'wb') as f:
    pickle.dump(backlog_hist, f)

#Plot results
x = np.arange(config.eval_freq, train_step + config.eval_freq*1, config.eval_freq)
y1 = reward_hist
y2 = TH_hist
y3 = ME_hist
y4 = backlog_hist

plt.figure()
plt.plot(x, y1, color='orange')
plt.title('Total reward per episode')
plt.xlabel('episodes')
plt.ylabel('reward')    
plt.show()

plt.figure()
plt.plot(x, y2, color='blue')
plt.title('Average throughput')
plt.xlabel('episodes')
plt.ylabel('TH')    
plt.show()
  
plt.figure()
plt.plot(x, y3, color='red')
plt.title('Average machine efficiency')
plt.xlabel('episodes')
plt.ylabel('ME')    
plt.show()    
    
plt.figure()
plt.plot(x, y4, color='green')
plt.title('Remaining backlog qty')
plt.xlabel('episodes')
plt.ylabel('backlog')    
plt.show()      
























    
    
    
    
    
    
    
    
    
    
    
    
    