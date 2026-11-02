import pandas as pd
import numpy as np
import toml
from spyder_kernels.utils.iofuncs import load_dictionary as ld

#Define model parameters
line_num = 3
layer_num = 6
product_num = 4
batch_size = 25
period_num = 21

#Training batch_qty
#batch_qty = [0,25,50,75,100,125,150,175,200]


#Testing batch_qty
batch_qty = [0,25,50,75,100,125,150,175,200,
             225,250,275,300,325,350,375,400]

machine_num1 = 5
machine_num2 = 10
machine_num3 = 8
max_machine_num = max(machine_num1, machine_num2, machine_num3)

#Machine num as input parameter to MILP model
machine_num = [range(machine_num1), 
               range(machine_num2), 
               range(machine_num3)]

#Create 2D and 3D matrix functions
def create_matrix(num_row, num_column):
    matrix = []
    for i in range(num_row):
        row = [0]*num_column
        matrix.append(row)
    return matrix  

def create_3D_matrix(num_dimen, num_row, num_column):
    matrix = [[0]  for n in range(0,num_dimen)]
    for n in range(0,num_dimen):
        add_val = np.zeros((num_row,num_column))
        matrix[n] = add_val
    return matrix

line_product = [[1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]]

line_product = np.array(line_product)


#Import and processing fixed parameters
travel_time_file = r'FILE_LOCATION\line_distance.xlsx' # <- Replace with your directory
travel_time = pd.read_excel(travel_time_file, index_col=0, sheet_name='Travel_time')


#Import processing time, setup time from spydata file
machine_time_file = ld(r'FILE_LOCATION\machine_time.spydata') # <- Replace with your directory
processing_time = machine_time_file[0][('processing_time')]
setup_time = machine_time_file[0][('setup_time')]



















            
            
            
            
            
            
            
            
            
            
            
            
            
            