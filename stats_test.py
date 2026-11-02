import pandas as pd
from scipy import stats
from data_preprocessing import create_matrix

data_file = r'C:\Python\SFL\Train_data\t_test.xlsx'
me_data = pd.read_excel(data_file, index_col=0, sheet_name='ME')
bl_data = pd.read_excel(data_file, index_col=0, sheet_name='BL')


me_table = pd.DataFrame(create_matrix(6, 4), columns = ["Product_num", "MNPL", "t_stats", "p_value"])
bl_table = pd.DataFrame(create_matrix(6, 4), columns = ["Product_num", "MNPL", "t_stats", "p_value"])

MNPL = 25
for i in range(0, 3):
    me_table.iloc[i,0] = 2
    me_table.iloc[i,1] = str(MNPL)+"%"
    bl_table.iloc[i,0] = 2
    bl_table.iloc[i,1] = str(MNPL)+"%"
    MNPL += 25
    
MNPL = 25
for i in range(3, 6):
    me_table.iloc[i,0] = 4
    me_table.iloc[i,1] = str(MNPL)+"%"
    bl_table.iloc[i,0] = 4
    bl_table.iloc[i,1] = str(MNPL)+"%"
    MNPL += 25    

def t_test(data_1, data_2):
    t_stats, p_value = stats.ttest_ind(data_1, data_2)
    return t_stats, p_value


sample_size = 10
s_idx = 0
e_idx = sample_size
for l in range(len(me_table)):
    SF_data = me_data["SF"].iloc[s_idx:e_idx]
    NSF_data = me_data["No SF"].iloc[s_idx:e_idx]
    me_table["t_stats"][l], me_table["p_value"][l] = t_test(SF_data, NSF_data)
    s_idx += sample_size
    e_idx += sample_size

sample_size = 10       
s_idx = 0
e_idx = sample_size
for l in range(len(bl_table)):
    SF_data = bl_data["SF"].iloc[s_idx:e_idx]
    NSF_data = bl_data["No SF"].iloc[s_idx:e_idx]
    bl_table["t_stats"][l], bl_table["p_value"][l] = t_test(SF_data, NSF_data)
    s_idx += sample_size
    e_idx += sample_size
       





















