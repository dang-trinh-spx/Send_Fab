#Function to calculate throughput
def return_throughput(production_qty, SF_decision):
    total_TH = sum(production_qty.values())
    if SF_decision:
        total_TH += sum(SF_decision.values())
    return total_TH

#Function to calculate ME
def return_ME(production_qty, 
              SF_decision, 
              machine_dedication, 
              line_num, 
              machine_num, 
              product_num, 
              layer_num, 
              processing_time):
    
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


#Function to calculate backlog
def return_remain_backlog(end_WIP, 
                          bl_table, 
                          line_num, 
                          product_num, 
                          layer_num):
    
    backlog_qty = 0
    for n in range(0, line_num):
        for i in range(0, product_num):
            for j in range(0, layer_num):
                bl_table[n][i].iloc[j,2] = end_WIP[n][i,j] #Input WIP
    
    ##Calculate lateness index
    for n in range(0, line_num):
        for i in range(0, product_num):
            for j in range(0, layer_num):
                if bl_table[n][i].iloc[j:,1].sum() > bl_table[n][i].iloc[j:,2].sum():
                    bl_table[n][i].iloc[j,4] += bl_table[n][i].iloc[j,3]
                else:
                    bl_table[n][i].iloc[j,4] = 0
    
    for n in range(0, line_num):
        for i in range(0, product_num):
            for j in range(0, layer_num):
                bl_table[n][i].iloc[j,5] = max(bl_table[n][i].iloc[j:,1].sum() -  bl_table[n][i].iloc[j:,2].sum(),0)
                
    for n in range(0, line_num):
        for i in range(0, product_num):
            for j in range(0, layer_num):
                if (bl_table[n][i].iloc[j,4]/(bl_table[n][i].iloc[j,3]+ 1e-10))*(1/3) >= 2: 
                   backlog_qty +=  bl_table[n][i].iloc[j,5]
    return backlog_qty








