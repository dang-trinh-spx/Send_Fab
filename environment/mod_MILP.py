import numpy as np 
import docplex.mp.model as cpx
from data_preprocessing import create_3D_matrix
from data_preprocessing import line_num, product_num, layer_num, batch_size, line_product

class MILP:
    def __init__(self, 
                 demand, 
                 begin_WIP,
                 sending_qty,
                 machine_dedication,
                 bl_table,
                 processing_time,
                 setup_time,
                 travel_time,
                 machine_num,
                 period,
                 line_product = line_product,
                 line_num = line_num,
                 product_num = product_num,
                 layer_num = layer_num,
                 batch_size = batch_size):    
        
        super().__init__()
        
        self.demand = demand
        self.begin_WIP = begin_WIP
        self.sending_qty = sending_qty
        self.machine_dedication = machine_dedication
        self.bl_table = bl_table
        self.processing_time = processing_time
        self.setup_time = setup_time
        self.travel_time = travel_time
        self.machine_num = machine_num
        self.line_num = line_num
        self.product_num = product_num
        self.layer_num = layer_num
        self.batch_size = batch_size
        self.line_product = line_product
        self.period = period
    
    #Function to update data to calculate backlog quantity
    def update_bl(self, bl_table):
        for n in range(0, self.line_num):
            for i in range(0, self.product_num):
                for j in range(0, self.layer_num):
                    self.bl_table[n][i].iloc[j,1] = self.demand[n][i,j] #Input demand
                    self.bl_table[n][i].iloc[j,2] = self.begin_WIP[n][i,j] #Input WIP
        
        #Calculate lateness (days) backlog = sum demand - sum WIP, if backlog > 0, lateness increases by layer flow time, otherwise lateness = 0
        for n in range(0, self.line_num):
            for i in range(0, self.product_num):
                for j in range(0, self.layer_num):
                    if self.bl_table[n][i].iloc[j:,1].sum() > self.bl_table[n][i].iloc[j:,2].sum():
                        self.bl_table[n][i].iloc[j,4] += self.bl_table[n][i].iloc[j,3]
                    else:
                        self.bl_table[n][i].iloc[j,4] = 0

        ##Calculate backlog quantity: sum demand - sum WIP
        for n in range(0, self.line_num):
            for i in range(0, self.product_num):
                for j in range(0, self.layer_num):
                    self.bl_table[n][i].iloc[j,5] = max(self.bl_table[n][i].iloc[j:,1].sum() -  self.bl_table[n][i].iloc[j:,2].sum(),0)
    
    
    #Function to pre-process input data to MILP model
    def MILP_data(self): 
        self.update_bl(self.bl_table)
            
        #Retrieve lateness index from backlog table
        self.lateness = create_3D_matrix(self.line_num, self.product_num, self.layer_num)
        for n in range(0, self.line_num):
            for i in range(0, self.product_num):
                for j in range(0, self.layer_num):
                    self.lateness[n][i,j] = self.bl_table[n][i].iloc[j,4]
        
        #Retrieve flow times from backlog table
        self.flow_time = create_3D_matrix(self.line_num, self.product_num, self.layer_num)
        for n in range(0, self.line_num):
            for i in range(0, self.product_num):
                for j in range(0, self.layer_num):
                    self.flow_time[n][i,j] = self.bl_table[n][i].iloc[j,3]
                    
        return self.lateness, self.flow_time, self.bl_table   
    
    
    def MILP_model(self):
        #Define sets, index range
        opt_model = cpx.Model(name = "SF")
        line_set = range(0,self.line_num)
        layer_set = range(0,self.layer_num)
        product_set = range(0,self.product_num)
        M = 100000
        
        #Import parameters
        l, c, self.bl_table = self.MILP_data()
        d = self.demand
        b = self.begin_WIP
        Q = self.sending_qty
        h = self.machine_dedication    
        T = self.travel_time
        p = self.processing_time
        ST = self.setup_time        
        machine_num = self.machine_num
        
        #Define index
        x = {(n,i,j,k): opt_model.continuous_var(lb=0, name = 'x({},{},{},{})'.format(n,i,j,k))
             for n in line_set for i in product_set for j in layer_set for k in machine_num[n]}
        
        y = {(n,i,j,k): opt_model.binary_var(name = 'y({},{},{},{})'.format(n,i,j,k))
             for n in line_set for i in product_set for j in layer_set for k in machine_num[n]}
        
        e = {(n,i,j): opt_model.continuous_var(lb = 0, name = 'e({},{},{})'.format(n,i,j))
             for n in line_set for i in product_set for j in layer_set}
            
        WPM = {(n,i,j): opt_model.continuous_var(lb = 0, name = 'WPT({},{},{})'.format(n,i,j))
             for n in line_set for i in product_set for j in layer_set}
        
        MAW = {(n,k): opt_model.continuous_var(lb = 0, name = 'MAW({},{})'.format(n,k))
             for n in line_set for k in machine_num[n]}
        
        AAW = {(n): opt_model.continuous_var(lb = 0, name = 'AAW({})'.format(n))
             for n in line_set}
            
        z = {(n,m,i,j,k): opt_model.binary_var(name = 'z({},{},{},{},{})'.format(n,m,i,j,k))
             for n in line_set for i in product_set for j in layer_set for m in line_set if m!=n for k in machine_num[m]}
        
        q = {(n,m,i,j,k): opt_model.continuous_var(lb = 0, name = 'q({},{},{},{},{})'.format(n,m,i,j,k))
             for n in line_set for i in product_set for j in layer_set for m in line_set if m!=n for k in machine_num[m]} 
            
        w0 = {(n,k): opt_model.binary_var()
              for n in line_set for k in machine_num[n]}
        
        w1 = {(n,k): opt_model.continuous_var(lb = 0)
              for n in line_set for k in machine_num[n]}
        
        w2 = {(n,k): opt_model.continuous_var(lb = 0)
              for n in line_set for k in machine_num[n]}
        
        L = {(n,i,j): opt_model.continuous_var(name = 'L({},{},{})'.format(n,i,j))
             for n in line_set for i in product_set for j in layer_set}
        
        L_linear = {(n,i,j): opt_model.binary_var(name = 'L_linear({},{},{})'.format(n,i,j))
             for n in line_set for i in product_set for j in layer_set}
        
        #Objective function:
        objective = (opt_model.sum(
        opt_model.sum(L[m,i,j] for i in product_set for j in layer_set)
        + 0.01*opt_model.sum(w1[m,k] + w2[m,k] for k in machine_num[m])
        + opt_model.sum(8 - opt_model.sum(x[m,i,j,k]*(p[m][i][j,k]/3600) for i in product_set for j in layer_set) 
                        - opt_model.sum(q[n,m,i,j,k]*(p[m][i][j,k]/3600) for i in product_set for j in layer_set for n in line_set if n!=m) for k in machine_num[m])*(3600/192) 
                        for m in line_set))
        
        #Set of constraints:
        #Linearize absolute function (load balancing)
        opt_model.add_constraints(w1[n,k] - w2[n,k] == MAW[n,k] - AAW[n] for n in line_set for k in machine_num[n])
        opt_model.add_constraints(w2[n,k] - w1[n,k] == AAW[n] - MAW[n,k] for n in line_set for k in machine_num[n])
        opt_model.add_constraints(w1[n,k] <= M*(1 - w0[n,k]) for n in line_set for k in machine_num[n])
        opt_model.add_constraints(w2[n,k] <= M*w0[n,k] for n in line_set for k in machine_num[n])
    
        #Calculate WPM
        opt_model.add_constraints(WPM[n,i,j] == e[n,i,j]/(np.sum(h[n][i][j,:]) + 1e-10)
                                  for n in line_set for i in product_set for j in layer_set)
        
        #Calculate MAW
        opt_model.add_constraints(MAW[n,k] == opt_model.sum(WPM[n,i,j]*h[n][i][j,k] for j in layer_set for i in product_set) 
                                  for n in line_set for k in machine_num[n])
                              
        #Calculate AAW
        opt_model.add_constraints(AAW[n] == opt_model.sum(MAW[n,k] for k in machine_num[n])/len(machine_num[n]) 
                                  for n in line_set)
            
        #Production assignment and production qty variable relation
        #opt_model.add_constraints(x[n,i,j,k] <= M*y[n,i,j,k] 
        #                          for n in line_set for i in product_set for j in layer_set for k in machine_num[n])
        
        #Calculate net processing time
        #opt_model.add_constraints(opt_model.sum(x[m,i,j,k]*(p[m][i][j,k]/3600) + y[m,i,j,k]*(ST[m][i][j,k]/60) for i in product_set for j in layer_set)
        #                          + opt_model.sum(q[n,m,i,j,k]*(p[m][i][j,k]/3600) + z[n,m,i,j,k]*(ST[m][i][j,k]/60) + z[n,m,i,j,k]*(T.iloc[n,m]/60) for n in line_set if n!=m for i in product_set for j in layer_set) 
        #                          <= 8 for m in line_set for k in machine_num[m])
        
        #Adding bounds
        #for n in line_set:
        #    for i in product_set:
        #        for j in layer_set:
        #            if b[n][i,j] < self.batch_size:
        #                opt_model.add_constraint(opt_model.sum(y[n,i,j,k] for k in machine_num[n]) == 0)
        
        #Production assignment and production qty variable relation (evaluation)
        opt_model.add_constraints(x[m,i,j,k] + q[n,m,i,j,k] <= M*y[m,i,j,k] 
                                  for m in line_set for i in product_set for j in layer_set for k in machine_num[m] for n in line_set if n!=m)
        
        opt_model.add_constraints(opt_model.sum(x[m,i,j,k]*(p[m][i][j,k]/3600) for i in product_set for j in layer_set)
                                + opt_model.sum(y[m,i,j,k]*(ST[m][i][j,k]/60) for i in product_set for j in layer_set)
                                + opt_model.sum(q[n,m,i,j,k]*(p[m][i][j,k]/3600) + z[n,m,i,j,k]*(T.iloc[n,m]/60) for n in line_set if n!=m for i in product_set for j in layer_set) 
                                <= 8 for m in line_set for k in machine_num[m])
                                          
        #WIP level restriction
        opt_model.add_constraints(opt_model.sum(x[n,i,j,k] for k in machine_num[n]) <= b[n][i,j] for n in line_set for i in product_set for j in layer_set)
            
        #Machine dedication on x 
        for n in line_set:
            for i in product_set:
                for j in layer_set:
                    for k in machine_num[n]:
                        if h[n][i][j,k] == 0:
                            if (n,i,j,k) in y:
                                opt_model.add_constraint(y[n,i,j,k] == 0)
                            
        #Machine dedication on q
        for m in line_set:
            for i in product_set:
                for j in layer_set:
                    for k in machine_num[m]:
                        for n in line_set:
                            if n!=m:
                                if h[m][i][j,k] == 0:
                                    if (n,m,i,j,k) in z:
                                        opt_model.add_constraint(z[n,m,i,j,k] == 0)
        
         
        #Calculate behind schedule qty (linearize)
        opt_model.add_constraints(L[n,i,j] <= M*L_linear[n,i,j] for n in line_set for i in product_set for j in layer_set)
        
        opt_model.add_constraints(L[n,i,j] >= 0 for n in line_set for i in product_set for j in layer_set)
        
        opt_model.add_constraints(L[n,i,j] >= opt_model.sum(d[n][i,p] for p in range(j, len(layer_set))) 
        - opt_model.sum(e[n,i,p] for p in range(j, len(layer_set))) for n in line_set for i in product_set for j in layer_set)
        
        #End WIP except layer 0
        opt_model.add_constraints(e[n,i,j] == b[n][i,j] - opt_model.sum(x[n,i,j,k] for k in machine_num[n]) + opt_model.sum(x[n,i,j-1,k] for k in machine_num[n]) 
        + opt_model.sum(q[n,m,i,j-1,k] for m in line_set if m!=n for k in machine_num[m]) 
        - opt_model.sum(q[n,m,i,j,k] for m in line_set if m!=n for k in machine_num[m]) for n in line_set for i in product_set for j in layer_set if j!=0)
                        
        #End WIP of sending line at layer 0
        opt_model.add_constraints(e[n,i,0] == b[n][i,0] - opt_model.sum(x[n,i,0,k] for k in machine_num[n]) + opt_model.sum(x[n,i,len(layer_set)-1,k] for k in machine_num[n]) 
        + opt_model.sum(q[n,m,i,len(layer_set)-1,k] for m in line_set if m!=n for k in machine_num[m]) 
        - opt_model.sum(q[n,m,i,0,k] for m in line_set if m!=n for k in machine_num[m]) for n in line_set for i in product_set)
                    
        #Send FAB decision and Send FAB qty relation
        opt_model.add_constraints(q[n,m,i,j,k] <= M*z[n,m,i,j,k] for n in line_set for i in product_set for j in layer_set for m in line_set if m!=n for k in machine_num[m])
        
        #Assign Send FAB quantity to receiving lines 
        opt_model.add_constraints(opt_model.sum(q[n,m,i,j,k] for m in line_set if m!=n for k in machine_num[m]) <= Q[n][i,j] for n in line_set for i in product_set for j in layer_set)
                                    
        opt_model.minimize(objective)
        opt_model.set_time_limit(5)
        opt_model.solve(clean_before_solve = True, log_output = False) #Adjust log_out = False to hide solving infos
        #print("-----Soltion-----")
        #print(opt_model.get_solve_details())
        #print("Period: ", self.period+1)
        #print("Objective value= ", objective_value)
        #print("Solving time= ", opt_model.solve_details.time)
        
        if opt_model.solution is not None:
            self.opt_x = opt_model.solution.get_value_dict(x, keep_zeros = True, precision = 1e-13)
            self.opt_y = opt_model.solution.get_value_dict(y, keep_zeros = True, precision = 1e-13)
            self.opt_e = opt_model.solution.get_value_dict(e, keep_zeros = True, precision = 1e-13)
            self.opt_q = opt_model.solution.get_value_dict(q, keep_zeros = False, precision = 1e-13)
            self.opt_L = opt_model.solution.get_value_dict(L, keep_zeros = True, precision = 1e-13)
            #self.opt_WPM = opt_model.solution.get_value_dict(WPM,precision = 1e-13)
            #self.opt_MAW = opt_model.solution.get_value_dict(MAW,precision = 1e-13)
            #self.opt_AAW = opt_model.solution.get_value_dict(AAW,precision = 1e-13)
            #self.opt_z = opt_model.solution.get_value_dict(z,keep_zeros = False)
        else:
            self.opt_x = None
            self.opt_y = None
            self.opt_e = None
            self.opt_q = None
            self.opt_L = None
            #self.opt_MAW = None
            #self.opt_AAW = None 
            
        return self.opt_x, self.opt_y, self.opt_e, self.opt_q, self.opt_L, self.bl_table
    
    #Function to calculate reward 1 (backlog qty)                
    def current_backlog(self, agent_nb, backlog):
        current_backlog = [0 for _ in range(agent_nb)]
        for agent in range(agent_nb):
            current_backlog[agent] = sum(value for key, value in backlog.items() if key[0] == agent)
        return current_backlog
    
    #Function to calculate reward 2 (loss capacity)
    def loss_capacity(self, agent_nb, production_qty, SF_assignment, machine_dedication):
        loss_capacity = [0 for _ in range(agent_nb)]
        for n in range(agent_nb):
            line_loss = 0
            for k in (self.machine_num[n]):
                processed_WIP = 0
                for i in range(self.product_num):
                    for j in range(self.layer_num):
                        if (n,i,j,k) not in production_qty:
                            continue
                        else:
                            processed_WIP += production_qty[(n,i,j,k)]*(self.processing_time[n][i][j,k]/3600) 
                if SF_assignment:
                    for key, value in SF_assignment.items():
                        if key[1] == n and key[4] == k:
                            processed_WIP += value*(self.processing_time[key[1]][key[2]][key[3],key[4]]/3600)             
                if machine_dedication[n][i][j,k] != 0:             
                    machine_loss = (8-processed_WIP)*(3600/192) 
                else:
                    machine_loss = 0                   
                line_loss += machine_loss
            loss_capacity[n] = line_loss
        return loss_capacity
    
    #Function to calculate reward 2(new)
    def throughput(self, agent_nb, production_qty, SF_assignment):
        TH = [0 for _ in range(agent_nb)]
        for n in range(agent_nb):
            processed_WIP = 0
            for k in (self.machine_num[n]):
                for i in range(self.product_num):
                    for j in range(self.layer_num):
                        if (n,i,j,k) not in production_qty:
                            continue
                        else:
                            processed_WIP += production_qty[(n,i,j,k)]
            if SF_assignment:
                for key, value in SF_assignment.items():
                    if key[0] == n:
                        processed_WIP += value
            TH[n] = processed_WIP
        return TH
    
    
    #Update layer WIP_level (current shift end WIP = next shift begin WIP)                
    def update_parameters(self, prev_begin_WIP, end_WIP):
        for n in range(self.line_num):
            for i in range(self.product_num):
                for j in range(self.layer_num):
                    if end_WIP.get((n,i,j)):
                        prev_begin_WIP[n][i,j] = end_WIP[(n,i,j)]
                    else:
                        prev_begin_WIP[n][i,j] = 0
                            

                    
                
    
    
            
            
           
            























