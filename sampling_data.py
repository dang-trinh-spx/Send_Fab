import numpy as np
import random
import pandas as pd
from data_preprocessing import period_num, line_num, product_num, layer_num, machine_num1, machine_num2, machine_num3, line_product
from data_preprocessing import create_3D_matrix, create_matrix

#Define datasets parameters
#Generate initial WIP level
iWIP_dist1 = [1000,1500]
iWIP_dist2 = [200,500]
iWIP_dist3 = [200,500]
iWIP_dist = [iWIP_dist1, 
             iWIP_dist2, 
             iWIP_dist3]

#Generate MNPL
MNPL_dist1 = [2,3]
MNPL_dist2 = [5,5]
MNPL_dist3 = [4,4]
MNPL_dist = [MNPL_dist1, 
             MNPL_dist2, 
             MNPL_dist3]

#Genrate demand
demand_dist1 = [2000,2500]
demand_dist2 = [200,500]
demand_dist3 = [200,500]
demand_dist = [demand_dist1, 
               demand_dist2, 
               demand_dist3]

#Flow time distribution
flowtime_mean = 1
flowtime_std = 0.25

class data_sampling:
    def __init__(self,
                 period_num,
                 line_num,
                 product_num,
                 layer_num,
                 demand_dist,
                 flowtime_mean,
                 flowtime_std,
                 iWIP_dist,
                 machine_num1,
                 machine_num2,
                 machine_num3,
                 MNPL_1,
                 MNPL_2,
                 MNPL_3):
        super().__init__()
        
        #Process input data
        self.period_num = period_num
        self.line_num = line_num
        self.product_num = product_num
        self.layer_num = layer_num
        self.demand_dist = demand_dist
        self.flowtime_mean = flowtime_mean
        self.flowtime_std = flowtime_std
        self.iWIP_dist = iWIP_dist
        self.machine_num1 = machine_num1
        self.machine_num2 = machine_num2
        self.machine_num3 = machine_num3
        self.MNPL_1 = MNPL_1
        self.MNPL_2 = MNPL_2
        self.MNPL_3 = MNPL_3
        self.line_product = line_product
        
        #Create generate data sets
        self.weekly_demand = create_matrix(self.line_num, self.product_num)
        self.weekly_demand = pd.DataFrame(self.weekly_demand, index = ["Line_1", "Line_2", "Line_3"] , columns = ["Product_1", "Product_2", "Product_3", "Product_4"])
        
        self.flow_time = create_3D_matrix(self.line_num, self.product_num, self.layer_num) 
        
        self.layer_demand = create_3D_matrix(self.line_num, self.product_num, self.layer_num)
        
        self.initial_WIP = create_3D_matrix(self.line_num, self.product_num, self.layer_num)
        
        self.machines_set = pd.DataFrame(columns = ['Line', 'Machines_number', 'MNPL'])
        
        self.machine_num = dict(zip([_ for _ in range(self.line_num)], 
                                    [range(self.machine_num1), 
                                     range(self.machine_num2), 
                                     range(self.machine_num3)]
                                    ))        
        
        self.machine_dedication = [[] for p in range(self.period_num)]
        for p in range(self.period_num):
            self.machine_dedication[p] = [[] for i in range(self.line_num)]
            for n in range(self.line_num):
                self.machine_dedication[p][n] = create_3D_matrix(self.product_num, self.layer_num, len(self.machine_num[n]))
        
    def generate_demand(self):
        for n in range(self.line_num):
            for i in range(self.product_num):
                #if self.line_product[n,i] != 0:
                #    if n != 1 and n != 4:  
                        #self.weekly_demand.iloc[n,i] = round(np.random.normal(self.demand_dist[0][0], self.demand_dist[0][1]))
                    #else:
                        #self.weekly_demand.iloc[n,i] = round(np.random.normal(self.demand_dist[1][0], self.demand_dist[1][1]))
                #else:
                    #continue
                    
                self.weekly_demand.iloc[n,i] = round(np.random.uniform(self.demand_dist[n][0], self.demand_dist[n][1]),0)
                
        return self.weekly_demand
                    
    def generate_flowtime(self):
        for n in range(self.line_num):
            for i in range(self.product_num):
                if self.line_product[n,i] != 0:
                    for j in range(self.layer_num):
                        while self.flow_time[n][i,j] <= 0.5 or self.flow_time[n][i,j] >= 2:
                            self.flow_time[n][i,j] = round(np.random.normal(self.flowtime_mean, self.flowtime_std),1)                  
                else:
                    continue
        return self.flow_time
    
    def generate_layer_demand(self):
        for n in range(self.line_num):
            for i in range(self.product_num):
                for j in range(self.layer_num):
                    self.layer_demand[n][i,j] = (self.weekly_demand.iloc[n,i] * self.flow_time[n][i][j]) / 7
        return self.layer_demand
    
    def generate_iWIP(self):
        for n in range(self.line_num):
            for i in range(self.product_num):
                #if self.line_product[n,i] != 0:
                #    if n != 1 and n != 4: 
                #           self.initial_WIP[n][i,0] = round(np.random.uniform(self.iWIP_dist[0][0], self.iWIP_dist[0][1]),0) #Generate initial WIP level for each product
                #    else:
                #        self.initial_WIP[n][i,0] = round(np.random.uniform(self.iWIP_dist[1][0], self.iWIP_dist[1][1]),0)
                #else:
                #    continue
                self.initial_WIP[n][i,0] = round(np.random.uniform(self.iWIP_dist[n][0], self.iWIP_dist[n][1]),0)
                
        return self.initial_WIP
   
    def generate_machine_info(self):
        parameters = [[1,self.machine_num1,self.MNPL_1],
                      [2,self.machine_num2,self.MNPL_2],
                      [3,self.machine_num3,self.MNPL_3]]
        for i in range(0, len(parameters)):
            self.machines_set.loc[len(self.machines_set)] = parameters[i]
        return self.machine_num, self.machines_set    
    
    def generate_machine_dedication(self):
        for p in range(self.period_num):
            for n in range(self.line_num):
                for i in range(self.product_num):
                    if self.line_product[n,i] != 0:
                        for j in range(self.layer_num):
                            while np.sum(self.machine_dedication[p][n][i][j,:]) not in self.machines_set.iloc[n]["MNPL"]:
                                for k in self.machine_num[n]:
                                    self.machine_dedication[p][n][i][j,k] = random.randint(0, 1)
                                    #if n == 1:
                                    #    if np.sum(self.machine_dedication[p][n][i][:,k]) == 0:
                                    #             self.machine_dedication[p][n][i][random.randint(0,self.layer_num-1),k] = 1
                    else:
                        continue   
                                             
        return self.machine_dedication
    
    def generate_machine_dedication_1(self):
        for p in range(self.period_num):
            for n in range(self.line_num):
                r1 = random.randint(0, product_num-1)
                r2 = random.randint(0, product_num-1)
                while r2 == r1:    
                    r2 = random.randint(0, product_num-1)
                if n==0:
                    for j in range(self.layer_num):
                        while (np.sum(self.machine_dedication[p][n][r1][j,:]) 
                               + np.sum(self.machine_dedication[p][n][r2][j,:]))  not in self.machines_set.iloc[n]["MNPL"]:
                            for k in self.machine_num[n]:
                                for i in range(self.product_num):
                                    self.machine_dedication[p][n][i][j,k] = random.randint(0, 1)
                        else:
                            continue   
                else:
                    for i in range(self.product_num):
                        for j in range(self.layer_num):
                            while np.sum(self.machine_dedication[p][n][i][j,:]) not in self.machines_set.iloc[n]["MNPL"]:
                                for k in self.machine_num[n]:
                                    self.machine_dedication[p][n][i][j,k] = random.randint(0, 1)
                                    #if np.sum(self.machine_dedication[p][n][i][:,k]) == 0:
                                        #self.machine_dedication[p][n][i][random.randint(0,self.layer_num-1),k] = 1
                            else:
                                continue
                                             
        return self.machine_dedication
    

#Generating data function
def initialization():  
    S = data_sampling(period_num, 
                      line_num, 
                      product_num, 
                      layer_num, 
                      demand_dist = demand_dist, 
                      flowtime_mean = flowtime_mean, 
                      flowtime_std = flowtime_std, 
                      iWIP_dist = iWIP_dist, 
                      machine_num1 = machine_num1, 
                      machine_num2 = machine_num2, 
                      machine_num3 = machine_num3, 
                      MNPL_1 = MNPL_dist[0], 
                      MNPL_2 = MNPL_dist[1], 
                      MNPL_3 = MNPL_dist[2])

    weekly_demand = S.generate_demand()
    flow_time = S.generate_flowtime()
    layer_demand = S.generate_layer_demand()
    initial_WIP = S.generate_iWIP()
    machine_num, machines_set = S.generate_machine_info()
    machine_dedication = S.generate_machine_dedication_1()

    data_batch = dict(zip(["weekly_demand", 
                                 "flow_time", 
                                 "layer_demand", 
                                 "initial_WIP", 
                                 "machine_num", 
                                 "machines_set", 
                                 "machine_dedication"],
                                [weekly_demand, 
                                 flow_time, 
                                 layer_demand, 
                                 initial_WIP, 
                                 machine_num, 
                                 machines_set, 
                                 machine_dedication]))   
    return data_batch


#Run initialization function
def gen_data(train_set, test_set):
    count = 0
    for i in range(len(train_set)):
        train_set[i] = initialization()
        test_set[i] = initialization()
        count += 1      

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Generate training and testing data sets
train_data_ls = [[] for _ in range(50)]
test_data_ls = [[] for _ in range(50)]

gen_data(train_data_ls, test_data_ls)
























