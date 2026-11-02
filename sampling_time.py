import numpy as np
from data_preprocessing import line_num, product_num, layer_num, machine_num, line_product
from data_preprocessing import create_3D_matrix


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Create function to generate machine time (processing_time and setup_time)
#Sampling processing time or setup time           
def sampling_time(array, mean, sd, ub, lb, machine_num):
    for n in range(0, line_num):
        for i in range(0, product_num):
            if line_product[n,i] != 0:
                for j in range(0, layer_num):
                    for k in machine_num[n]:
                        while True:
                            random_val = round(np.random.normal(mean,sd),0)
                            if lb <= random_val <= ub:
                                array[n][i][j,k] = random_val
                                break    
            else:
                continue

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#If generate input processing time and setup time, run the below code and save data
#Run when generating machine time
processing_time = [[] for _ in range(line_num)] 
setup_time = [[] for _ in range(line_num)] 

for n in range(line_num):
    processing_time[n] = create_3D_matrix(product_num, layer_num, len(machine_num[n])) 
    setup_time[n] = create_3D_matrix(product_num, layer_num, len(machine_num[n])) 

#Run the follwing lines to generate processing_time/setup_time
sampling_time(processing_time, 192, 10, 200, 180, machine_num)  
sampling_time(setup_time, 15, 5, 20, 10, machine_num) 