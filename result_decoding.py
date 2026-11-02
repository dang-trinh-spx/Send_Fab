import pickle
from spyder_kernels.utils.iofuncs import load_dictionary as ld


file_1 = r'C:\Python\SFL\MILP_result\opt_x.pkl'
#file_1 = r'C:\Python\Discrete_action\MILP_result\opt_x.pkl'
with open(file_1, 'rb') as file_1:
    production_qty = pickle.load(file_1)

file_2 = r'C:\Python\SFL\MILP_result\opt_e.pkl'
#file_2 = r'C:\Python\Discrete_action\MILP_result\opt_e.pkl'
with open(file_2, 'rb') as file_2:
    ending_WIP = pickle.load(file_2)
    
file_3 = r'C:\Python\SFL\MILP_result\opt_L.pkl'
#file_3 = r'C:\Python\Discrete_action\MILP_result\opt_L.pkl'
with open(file_3, 'rb') as file_3:
    backlog = pickle.load(file_3)
    
file_4 = r'C:\Python\SFL\MILP_result\opt_q.pkl'
#file_4 = r'C:\Python\Discrete_action\MILP_result\opt_q.pkl'
with open(file_4, 'rb') as file_4:
    SendFAB = pickle.load(file_4)
    

#Train and Test data list
input_data_dir = r'C:\Python\SFL\Train_data\train_data.spydata'

test_data_ls = ld(input_data_dir)
test_data_ls = test_data_ls[0][('train_data_ls')]
test_data = test_data_ls[0]    

file_6 = r'C:\Python\SFL\MAPPO_result\opt_x.pkl'
with open(file_6, 'rb') as file_6:
    RL_production_qty = pickle.load(file_6)

file_7 = r'C:\Python\SFL\MAPPO_result\opt_q.pkl'
with open(file_7, 'rb') as file_7:
    RL_SendFAB = pickle.load(file_7)

file_8 = r'C:\Python\SFL\MAPPO_result\opt_L.pkl'
with open(file_8, 'rb') as file_8:
    RL_backlog = pickle.load(file_8)
 
'''    
period = 9
MILP_throughput = sum(production_qty[period].values()) + sum(SendFAB[period].values())
RL_throughput = sum(RL_production_qty[period].values()) + sum(RL_SendFAB[period].values())


MILP_backlog = sum(backlog[period].values())
MAPPO_backlog = sum(RL_backlog[period].values()) 

print("MILP_TH: ", MILP_throughput)
print("RL_TH: ", RL_throughput)

#print("MILP_BL: ", MILP_backlog)
#print("RL_BL: ", MAPPO_backlog)  
'''    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    