from spyder_kernels.utils.iofuncs import load_dictionary as ld
from pathlib import Path

#Input Train data set
#train_data_dir = r'FILE_LOCATION' #<= Input train_data.spydata
#train_data_ls = ld(train_data_dir)[0][('data_ls')] #<= DO NOT CHANGE!
#eval_data = train_data_ls[0]

#Input Train data set
dir_1  = Path(r'C:\Python\SOP_SF\SF_project\datasets\two_products')
dir_2 = Path('train_data.spydata')
#test_data_dir = r'FILE_LOCATION' #<= Input _instance.spydata 
test_data_dir = dir_1 / dir_2 #<= Input _instance.spydata 
test_data_ls = ld(test_data_dir)[0][('data_ls')] #<= DO NOT CHANGE!


