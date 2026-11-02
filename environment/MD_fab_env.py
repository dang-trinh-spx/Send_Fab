import numpy as np
from gym.spaces import MultiDiscrete
from data_preprocessing import create_3D_matrix
from Env.mod_MILP import MILP
import copy 
import pandas as pd
from data_preprocessing import line_num, product_num, layer_num, batch_qty, period_num, processing_time, setup_time, travel_time

#Initialize parameters
line_num = line_num
product_num = product_num
layer_num = layer_num
batch_qty = batch_qty
period_num = period_num
processing_time = processing_time
setup_time = setup_time
travel_time = travel_time

#Define function to scale down value (logarithm sclaing)
class LogScale:
    def __init__(self):
        self.constant = 1 
        self.base = np.e   
             
    def log_scale(self,x):
        log_x = np.log(x + self.constant) / np.log(self.base)
        return log_x

class Standardization:
    def __init__(self):
        self.mean_val = None
        self.std_val = None
        
    def update(self, x):
        x = np.array(x)
        self.mean_val = np.mean(x, axis=0)
        self.std_val = np.std(x, axis=0)
        return self.mean_val, self.std_val
    
    def standard(self, x):
        if self.mean_val is None or self.std_val is None:
            raise ValueError("Standardization instance has not been updated with data yet.")
        x = np.array(x)
        standard_x = (x - self.mean_val) / (self.std_val + 1e-10)
        return standard_x

#Define environment class
class Fab_env:
    def __init__(self, agent_nb, input_data):
        super().__init__()
        
        #Define agents, state space, action space
        self.agent_nb = agent_nb
        
        self.agents = []
        for a in range(self.agent_nb):
            self.agents.append(a)
            
        self.observation_space = [np.zeros([12,product_num,layer_num]) for _ in range(self.agent_nb)]
        self.global_state = np.zeros([12,product_num*self.agent_nb,layer_num])
        #self.observation_space = [np.zeros([2,product_num,layer_num]) for _ in range(self.agent_nb)]
        
        self.action_space = [MultiDiscrete([product_num,layer_num,len(batch_qty)]) for _ in range(self.agent_nb)]
        
        #Extract data from input data
        self.flow_time = input_data['flow_time']
        self.initial_WIP = input_data['initial_WIP']
        self.demand = input_data['layer_demand']
        self.machine_dedication = input_data['machine_dedication']
        self.machine_num = input_data['machine_num']
        self.machines_set = input_data['machines_set'] 
        
        #Create begin WIP (at the begining of every shift)
        self.begin_WIP = copy.deepcopy(self.initial_WIP)
        
        #Create initial backlog    
        self.initial_backlog = create_3D_matrix(line_num, product_num, layer_num)
        for n in range(line_num):
            for i in range(product_num):
                for j in range(layer_num):
                    self.initial_backlog[n][i,j] = max((np.sum(self.demand[n][i,j:]) - np.sum(self.initial_WIP[n][i,j:])),0)
        
        #Create initial machine dedication (re-format to match the input_dims)
        self.initial_machine_dedication = [[] for _ in range(line_num)]  
        for n in range(line_num):
            self.initial_machine_dedication[n] = create_3D_matrix(self.machines_set['Machines_number'].max(), product_num, layer_num)   
        for n in range(line_num):
            for k in self.machine_num[n]:
                for i in range(product_num):
                    for j in range(layer_num):
                        self.initial_machine_dedication[n][k][i,j] = self.machine_dedication[0][n][i][j,k]
        
        #Create backlog table
        self.bl_table = [[] for i in range(0, line_num)]  
        for n in range(0, line_num):
            add_val = create_3D_matrix(product_num, layer_num, 6)
            self.bl_table[n] = add_val   
        
        for n in range(0, line_num):
            for i in range(0, product_num):
                self.bl_table[n][i] = pd.DataFrame(self.bl_table[n][i], columns = ['Layer', 'Demand','WIP', 'Flow times', 'Lateness', 'Backlog_qty'])
                
        #Format layer index in backlog table                              
        for n in range(0, line_num): 
            for i in range(0, product_num):
                for j in range(0, layer_num):
                    self.bl_table[n][i].iloc[j,0] = j + 1
        
        #Input flowtime to backlog table
        for n in range(0, line_num):
            for i in range(0, product_num):
                for j in range(0, layer_num):
                    self.bl_table[n][i].iloc[j,3] = self.flow_time[n][i,j]
        
    def observation_space(self, agent):
        return self.observation_space[agent]
    
    def global_state(self):
        return self.global_state
    
    def action_space(self, agent):
        return self.action_space[agent]
     
    #Function to re-formate machine dedication data to macth onbservation space
    def reshape_machine_dedication(self, step_nb):
        current_machine_dedication = [[] for _ in range(line_num)]
        for n in range(line_num):
            current_machine_dedication[n] = create_3D_matrix(self.machines_set['Machines_number'].max(), product_num, layer_num)    
        for n in range(line_num):
            for k in self.machine_num[n]:
                for i in range(product_num):
                    for j in range(layer_num):
                        current_machine_dedication[n][k][i,j] = self.machine_dedication[step_nb][n][i][j,k]
        return current_machine_dedication
    
    #Define step function 
    def step(self, action_space):
        #0 Initiate variable to check MILP feasibility
        feasible = 1
        
        #1 Initiate parameters
        rewards = [0 for _ in range(self.agent_nb)]
        sending_qty = create_3D_matrix(line_num, product_num, layer_num)
        WIP_penalty = [0 for _ in range(self.agent_nb)]
        
        #1.1 sending_qty is the action of all the agents, form a input data Q[n,i,j] to the MILP model
        for a in range(self.agent_nb):
            if self.begin_WIP[a][action_space[a][0], action_space[a][1]] == 0:
                WIP_penalty[a] += 500
                sending_qty[a][action_space[a][0],action_space[a][1]] = batch_qty[0]
            else:
                if self.begin_WIP[a][action_space[a][0], action_space[a][1]] < batch_qty[action_space[a][2]]:
                    WIP_penalty[a] += 500
                    sending_qty[a][action_space[a][0],action_space[a][1]] = batch_qty[0]
                else:    
                    sending_qty[a][action_space[a][0],action_space[a][1]] = batch_qty[action_space[a][2]]
        
        #2 Run MILP model
        #2.1 Input data to MILP model
        opt_model = MILP(demand = self.demand,
                         begin_WIP = self.begin_WIP,
                         sending_qty = sending_qty,
                         machine_dedication = self.machine_dedication[self.step_nb],
                         bl_table = self.bl_table,
                         processing_time = processing_time,
                         setup_time = setup_time,
                         travel_time = travel_time,
                         machine_num = self.machine_num,
                         period = self.step_nb)
        
        shift_machine_dedication = self.machine_dedication[self.step_nb]
                
        #2.2 Run model
        production_qty, assignment, end_WIP, SF_decision, backlog, bl_table = opt_model.MILP_model()

        if production_qty != None:
            #3 Calculate reward based on MILP results
            #3.1 Calculate loss capacity (reward 1)
            current_backlog = opt_model.current_backlog(self.agent_nb, backlog)
                        
            #4 Update step_num
            self.step_nb += 1
            
            #4.1 Terminate if reaches maximum shift number (done = True)
            if self.step_nb == period_num:
                self.done = True
                current_machine_dedication = self.reshape_machine_dedication(self.step_nb-1) ##Update machine dedication for next shift (= current shift if it is the final shift)
            else:
                current_machine_dedication = self.reshape_machine_dedication(self.step_nb) ##Update machine dedication for next shift
            
            #5 Update new observations: 3 components -> Updated WIP level, Backlog, Machine dedication. 8 = Begin WIP layer + Backlog layer + Maximum Machines layers
            current_obs = [np.zeros([12,product_num,layer_num]) for _ in range(self.agent_nb)]
            current_state = np.zeros([12,product_num*self.agent_nb,layer_num])
            
            #Update current observations
            for a in range(self.agent_nb):
                for i in range(product_num):
                    for j in range(layer_num):
                        if end_WIP.get((a,i,j)):
                            current_obs[a][0][i,j] = end_WIP[(a,i,j)]   ## 1st observation: WIP level
                        else:
                            current_obs[a][0][i,j] = 0
                            
                        if backlog.get((a,i,j)):
                            current_obs[a][1][i,j] = backlog[(a,i,j)] ## 2nd observation: backlog
                        else:
                            current_obs[a][1][i,j] = 0
                
                current_obs[a][2:] = current_machine_dedication[a] ## 3rd observation: machine dedication (next shift)
                
            #Update current states
            for l in range(len(current_state)):
                down = 0
                for a in range(self.agent_nb):
                    for i in range(product_num):
                        for j in range(layer_num):   
                            current_state[0][i+down,j] = current_obs[a][0][i,j]
                            current_state[1][i+down,j] = current_obs[a][1][i,j]
                            if l != 0 and l != 1:
                                current_state[l][i+down,j] = current_obs[a][l][i,j]
                    down += product_num
            
            for a in range(self.agent_nb):
                #Define reward
                #rewards[a] = -(sum(WIP_penalty) + sum(current_backlog)) #MAPPO reward
                rewards[a] = -(WIP_penalty[a] + current_backlog[a]) #IPPO reward

            
            #7 Update begin WIP of next shift (= end WIP of this shift)
            opt_model.update_parameters(self.begin_WIP, end_WIP)
        
        else: 
            feasible = 0
            current_obs = None
            current_state = None
            rewards = None
            self.done = None
            self.step_nb = None
            bl_table = None
            production_qty = None
            backlog = None
            sending_qty = None
            SF_decision = None
            self.begin_WIP = None
            shift_machine_dedication = None

                 
        return  feasible,\
                current_obs,\
                current_state,\
                rewards,\
                self.done,\
                self.step_nb,\
                bl_table,\
                production_qty,\
                backlog, sending_qty,\
                SF_decision,\
                self.begin_WIP,\
                shift_machine_dedication


        
    def reset(self, seed=None):
        self.step_nb = 0
        
        #Reset observation, reward, done, info
        for a in range(self.agent_nb):
            self.observation_space[a][0] = copy.deepcopy(self.initial_WIP[a]) 
            self.observation_space[a][1] = copy.deepcopy(self.initial_backlog[a]) 
            self.observation_space[a][2:] = copy.deepcopy(self.initial_machine_dedication[a]) 
                        
        self.done = False
        
        #Reset begin WIP
        self.begin_WIP = copy.deepcopy(self.initial_WIP)
        
        #Reset global state
        for l in range(len(self.global_state)):
            down = 0
            for a in range(self.agent_nb):
                for i in range(product_num):
                    for j in range(layer_num):   
                        self.global_state[0][i+down,j] = self.observation_space[a][0][i,j]
                        self.global_state[1][i+down,j] = self.observation_space[a][1][i,j]
                        if l != 0 and l != 1:
                            self.global_state[l][i+down,j] = self.observation_space[a][l][i,j]
                down += product_num
                
        return self.global_state, self.observation_space
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        