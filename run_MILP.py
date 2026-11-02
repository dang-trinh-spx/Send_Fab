import numpy as np
from data_preprocessing import create_3D_matrix
from MILP.cplex_MILP import MILP
import copy 
import pandas as pd
import pickle

from data_preprocessing import line_num, product_num, layer_num, batch_qty, period_num, machine_num
from data_preprocessing import processing_time, setup_time, travel_time
from spyder_kernels.utils.iofuncs import load_dictionary as ld


#Initialize parameters
line_num = line_num
product_num = product_num
layer_num = layer_num
batch_qty = batch_qty
period_num = period_num
processing_time = processing_time
setup_time = setup_time
travel_time = travel_time

#Train and Test data list
input_data_dir = r'C:\Python\SFL\Train_data\train_data.spydata'

test_data_ls = ld(input_data_dir)
test_data_ls = test_data_ls[0][('train_data_ls')]
test_data = test_data_ls[0]

#Define function to scale down value (logarithm sclaing)
class LogScale:
    def __init__(self):
        self.constant = 1 
        self.base = np.e   
             
    def log_scale(self,x):
        log_x = np.log(x + self.constant) / np.log(self.base)
        return log_x


class rollout:
    def __init__(self, input_data):
        super().__init__()
        
        #Extract data for input data dict
        self.flow_time = input_data['flow_time']
        self.demand = input_data['layer_demand']
        self.initial_WIP = input_data['initial_WIP']
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
        
        #Create initial machine dedication (format to match the input_dims)
        self.initial_machine_dedication = [[] for _ in range(line_num)]  
        for n in range(line_num):
            self.initial_machine_dedication[n] = create_3D_matrix(self.machines_set['Machines_number'].max(), product_num, layer_num)   
        for n in range(line_num):
            for k in self.machine_num[n]:
                for i in range(product_num):
                    for j in range(layer_num):
                        self.initial_machine_dedication[n][k][i,j] = self.machine_dedication[0][n][i][j,k]
        
        #Create UPQ table
        self.UPQ_table = [[] for i in range(0, line_num)]  
        for n in range(0, line_num):
            add_val = create_3D_matrix(product_num, layer_num, 6)
            self.UPQ_table[n] = add_val   
        
        for n in range(0, line_num):
            for i in range(0, product_num):
                self.UPQ_table[n][i] = pd.DataFrame(self.UPQ_table[n][i], 
                                                    columns = ['Layer', 'Demand','WIP', 'Flow times', 'Lateness', 'Backlog_qty'])
        #UPQ table layer number                              
        for n in range(0, line_num): 
            for i in range(0, product_num):
                for j in range(0, layer_num):
                    self.UPQ_table[n][i].iloc[j,0] = j + 1
        
        #UPQ table flow time
        for n in range(0, line_num):
            for i in range(0, product_num):
                for j in range(0, layer_num):
                    self.UPQ_table[n][i].iloc[j,3] = self.flow_time[n][i,j]
        
    def solving(self, period):
        share_reward = 0
        
        #Initiate MILP model
        opt_model = MILP(demand = self.demand,
                         begin_WIP = self.begin_WIP,
                         machine_dedication = self.machine_dedication[period],
                         UPQ_table = self.UPQ_table,
                         processing_time = processing_time,
                         setup_time = setup_time,
                         travel_time = travel_time,
                         machine_num = self.machine_num,
                         period = period)
                
        #Run MILP model
        production_qty, end_WIP, SF_decision, backlog, solving_time, self.UPQ_table, shift_machine_dedication = opt_model.MILP_model() 
        
        #Update begin WIP and initial processing index
        opt_model.update_parameters(self.begin_WIP, end_WIP)
        
        return production_qty,\
                backlog,\
                SF_decision,\
                solving_time,\
                self.begin_WIP,\
                self.UPQ_table,\
                shift_machine_dedication
                
def return_ME_old(production_qty, SF_decision):
    #Return average ME (col = 4)
    global_ME = 0
    for n in range(line_num):
        for k in machine_num[n]:
            local_ME = [0 for _ in range(len(machine_num[n]))]
            for i in range(product_num):
                for j in range(layer_num):
                    if (n,i,j,k) in production_qty:
                        local_ME[k] += (production_qty[(n,i,j,k)]*(processing_time[n][i][j,k]/60/60))/8
                        for m in range(line_num):
                            if m!= n:    
                                if (m,n,i,j,k) in SF_decision:
                                    local_ME[k] += (SF_decision[(m,n,i,j,k)]*(processing_time[n][i][j,k]/60/60))/8
                                    pass
                        pass
            avg_local_ME = sum(local_ME)/len(machine_num[n])
            global_ME += avg_local_ME
    global_ME = round(global_ME/line_num,2)
    return global_ME     

#Function to calculate ME
def return_ME(production_qty, SF_decision, machine_dedication):
    #Return average ME (col = 4)
    global_ME = 0
    for n in range(line_num):
        active_machine = [0 for _ in range(len(machine_num[n]))]
        local_ME = [0 for _ in range(len(machine_num[n]))]
        for k in machine_num[n]:
            for i in range(product_num):
                for j in range(layer_num):
                    if (n,i,j,k) in production_qty:
                        local_ME[k] += (production_qty[(n,i,j,k)]*(processing_time[n][i][j,k]/60/60))/8
                        for m in range(line_num):
                            if m!= n:    
                                if (m,n,i,j,k) in SF_decision:
                                    local_ME[k] += (SF_decision[(m,n,i,j,k)]*(processing_time[n][i][j,k]/60/60))/8
                                    pass
                        pass
                    
                    if machine_dedication[n][i][j,k] != 0:
                        active_machine[k] = 1
                    
        avg_local_ME = sum(local_ME)/sum(active_machine)
        global_ME += avg_local_ME
    global_ME = global_ME/line_num
    return global_ME    

def return_remain_backlog(begin_WIP, UPQ_table):
    backlog_qty = 0
    for n in range(0, line_num):
        for i in range(0, product_num):
            for j in range(0, layer_num):
                UPQ_table[n][i].iloc[j,2] = begin_WIP[n][i,j] #Input WIP
    
    ##Calculate lateness index
    for n in range(0, line_num):
        for i in range(0, product_num):
            for j in range(0, layer_num):
                if UPQ_table[n][i].iloc[j:,1].sum() > UPQ_table[n][i].iloc[j:,2].sum():
                    UPQ_table[n][i].iloc[j,4] += UPQ_table[n][i].iloc[j,3]
                else:
                    UPQ_table[n][i].iloc[j,4] = 0
    
    for n in range(0, line_num):
        for i in range(0, product_num):
            for j in range(0, layer_num):
                UPQ_table[n][i].iloc[j,5] = max(UPQ_table[n][i].iloc[j:,1].sum() -  UPQ_table[n][i].iloc[j:,2].sum(),0)
                
    for n in range(0, line_num):
        for i in range(0, product_num):
            for j in range(0, layer_num):
                if (UPQ_table[n][i].iloc[j,4]/UPQ_table[n][i].iloc[j,3] + 1e-10)*(1/3) >= 2: 
                   backlog_qty +=  UPQ_table[n][i].iloc[j,5]
    return backlog_qty

#Run MILP model
M = rollout(input_data = test_data)
    
TH_ls = []
ME_ls = []
SF_ls = []
cpu_time = []
bl_list = []
EWIP_list = []
production_list = []

for period in range(0, period_num):
    shift_TH = 0
    production_qty, backlog, SF_decision, solving_time, begin_WIP, UPQ_table, shift_machine_dedication = M.solving(period)
    SF_ls.append(SF_decision)
    bl_list.append(backlog)
    EWIP_list.append(begin_WIP)
    
    #Calculate total TH of this shift
    shift_TH += sum(production_qty.values())
    if SF_decision:
        shift_TH += sum(SF_decision.values())
    TH_ls.append(shift_TH)

    #Calculate average ME
    shift_ME = return_ME(production_qty, SF_decision, shift_machine_dedication)
    ME_ls.append(shift_ME)    
    cpu_time.append(solving_time)
    production_list.append(production_qty)

final_WIP = begin_WIP
        
avg_TH = np.mean(TH_ls)
avg_ME = np.mean(ME_ls)
remain_backlog = return_remain_backlog(final_WIP, UPQ_table)
    
print("Test_avg_TH: ", avg_TH)
print("Test_avg_ME: ", avg_ME)
print("Test_remain_backlog: ", remain_backlog)
print("CPU_time: ", cpu_time)

with open('MILP_result/opt_x.pkl', 'wb') as f:
    pickle.dump(production_list, f)        
        
with open('MILP_result/opt_q.pkl', 'wb') as f:
    pickle.dump(SF_ls, f)    

with open('MILP_result/opt_L.pkl', 'wb') as f:
    pickle.dump(bl_list, f)            
        
with open('MILP_result/opt_e.pkl', 'wb') as f:
    pickle.dump(EWIP_list, f)    

with open('MILP_result/ME.pkl', 'wb') as f:
    pickle.dump(ME_ls, f)    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        