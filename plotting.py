import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy.signal import savgol_filter
from matplotlib.ticker import MaxNLocator

opt_th = 3092
opt_bl = 4385
opt_me = 89.66

plot_opt_th = [[] for _ in range(250)]
plot_opt_bl = [[] for _ in range(250)]
plot_opt_me = [[] for _ in range(250)]

for i in range(250):
    plot_opt_th[i].append(opt_th)
    plot_opt_bl[i].append(opt_bl)
    plot_opt_me[i].append(opt_me)


#1. Load result file using pickle
#1.1 Load new MAPPO results
b84_reward_dir = r'C:\Python\Main\Eval_SF\trained_MAPPO\B84\results\reward.pkl'
with open(b84_reward_dir, 'rb') as file_1:
    b84_reward = pickle.load(file_1)

b84_th_dir = r'C:\Python\Main\Eval_SF\trained_MAPPO\B84\results\th.pkl'
with open(b84_th_dir, 'rb') as file_2:
    b84_th = pickle.load(file_2)
    
b84_bl_dir = r'C:\Python\Main\Eval_SF\trained_MAPPO\B84\results\bl.pkl'
with open(b84_bl_dir, 'rb') as file_3:
    b84_bl = pickle.load(file_3)
    
b84_me_dir = r'C:\Python\Main\Eval_SF\trained_MAPPO\B84\results\me.pkl'
with open(b84_me_dir, 'rb') as file_4:
    b84_me = pickle.load(file_4)
    
#----------------------------------------------------------------------------------------------------------------

#1.2 Load old MAPPO results
b105_reward_dir = r'C:\Python\Main\Eval_SF\trained_MAPPO\B105\results\reward.pkl'
with open(b105_reward_dir, 'rb') as file_5:
    b105_reward = pickle.load(file_5)

b105_th_dir = r'C:\Python\Main\Eval_SF\trained_MAPPO\B105\results\th.pkl'
with open(b105_th_dir, 'rb') as file_6:
    b105_th = pickle.load(file_6)
    
b105_bl_dir = r'C:\Python\Main\Eval_SF\trained_MAPPO\B105\results\bl.pkl'
with open(b105_bl_dir, 'rb') as file_7:
    b105_bl = pickle.load(file_7)
    
b105_me_dir = r'C:\Python\Main\Eval_SF\trained_MAPPO\B105\results\me.pkl'
with open(b105_me_dir, 'rb') as file_8:
    b105_me = pickle.load(file_8)
    
#----------------------------------------------------------------------------------------------------------------    
    
#1.3 Load MAPPO results
b140_reward_dir = r'C:\Python\Main\Eval_SF\trained_MAPPO\B140\results\reward.pkl'
with open(b140_reward_dir, 'rb') as file_9:
    b140_reward = pickle.load(file_9)

b140_th_dir = r'C:\Python\Main\Eval_SF\trained_MAPPO\B140\results\th.pkl'
with open(b140_th_dir, 'rb') as file_10:
    b140_th = pickle.load(file_10)
    
b140_bl_dir = r'C:\Python\Main\Eval_SF\trained_MAPPO\B140\results\bl.pkl'
with open(b140_bl_dir, 'rb') as file_11:
    b140_bl = pickle.load(file_11)

b140_me_dir = r'C:\Python\Main\Eval_SF\trained_MAPPO\B140\results\me.pkl'
with open(b140_me_dir, 'rb') as file_12:
    b140_me = pickle.load(file_12)


#2. Define parameters
window_size = 80
eval_freq = 20
train_episode = 5000
alpha = 2/(window_size+1)
sigma = 30

#3. Define function for smoothing and computing deviation
def smooth_gaussian(data, sigma):
    smooth_data = copy.deepcopy(data)
    
    #Gaussian 
    smooth_data = gaussian_filter(smooth_data, sigma)
    
    return smooth_data
    
def compute_dev(smooth_data, window_size):
    half_window = window_size // 2
    deviation = np.zeros_like(smooth_data)
        
    for i in range(len(smooth_data)):
        start = max(0, i - half_window)
        end = min(len(smooth_data), i + half_window + 1)
        window_data = smooth_data[start:end]
        deviation[i] = np.std(window_data)
    return deviation

##DRAW BACKLOG (window_size = 30, sigma = 15)
#4. Calculate smooth_data and deviation
def draw_bl(bl1, bl2, bl3, window_size, sigma):
    smooth_bl1 = smooth_gaussian(bl1, sigma)
    smooth_bl1_dev = compute_dev(smooth_bl1, window_size)
    
    smooth_bl2 = smooth_gaussian(bl2, sigma)
    smooth_bl2_dev = compute_dev(smooth_bl2, window_size)
    
    smooth_bl3 = smooth_gaussian(bl3, sigma)
    smooth_bl3_dev = compute_dev(smooth_bl3, window_size)
    
    #5. Plot Backlog quantity
    #Define x-axis
    x = np.arange(eval_freq, train_episode + eval_freq, eval_freq)
    
    #Define mappo axis (y1, y2)
    y1 = smooth_bl1
    y2 = smooth_bl1_dev
    
    #Define mappo axis (y3, y4)
    y3 = smooth_bl2
    y4 = smooth_bl2_dev
    
    #Define mappo axis (y5, y6)
    y5 = smooth_bl3
    y6 = smooth_bl3_dev
    
    #plt.figure(figsize=(3,6))
    plt.plot(x, y1, label = 'b84', color='blue', alpha = 1)
    plt.plot(x, y3, label = 'b105', color='red', alpha = 1)
    plt.plot(x, y5, label = 'b140', color='green', alpha = 1)
    
    plt.fill_between(x, y1 - y2, y1 + y2, color = 'blue', alpha = 0.1)
    plt.fill_between(x, y3 - y4, y3 + y4, color = 'red', alpha = 0.1)
    plt.fill_between(x, y5 - y6, y5 + y6, color = 'green', alpha = 0.1)
    
    plt.xlabel('Episodes', fontsize=12, labelpad=10)
    plt.ylabel('Backlog quantity (wafers)', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    #plt.ylim(4000,9000)
    plt.legend()
    plt.grid(True)
    plt.savefig(r"\batchsize_backlog.png", dpi=300, bbox_inches='tight') #Input the saving directory
    plt.show()


def draw_th(th1, th2, th3, window_size, sigma):
    ##DRAW THROUGHPUT (window_size = 50, sigma = 25)
    #4. Calculate smooth_data and deviation
    smooth_th1 = smooth_gaussian(th1, sigma)
    smooth_th1_dev = compute_dev(smooth_th1, window_size)
    
    smooth_th2 = smooth_gaussian(th2, sigma)
    smooth_th2_dev = compute_dev(smooth_th2, window_size)
    
    smooth_th3 = smooth_gaussian(th3, sigma)
    smooth_th3_dev = compute_dev(smooth_th3, window_size)
    
    #5. Plot Backlog quantity
    #Define x-axis
    x = np.arange(eval_freq, train_episode + eval_freq, eval_freq)
    
    #Define mappo axis (y1, y2)
    y1 = smooth_th1
    y2 = smooth_th1_dev
    
    #Define ippo axis (y3, y4)
    y3 = smooth_th2
    y4 = smooth_th2_dev
    
    #Define ippo axis (y3, y4)
    y5 = smooth_th3
    y6 = smooth_th3_dev
    
    #plt.figure(figsize=(12,10))
    plt.plot(x, y1, label = 'b84', color='blue', alpha = 1)
    plt.plot(x, y3, label = 'b105', color='red', alpha = 1)
    plt.plot(x, y5, label = 'b140', color='green', alpha = 1)
    
    plt.fill_between(x, y1 - y2, y1 + y2, color = 'blue', alpha = 0.1)
    plt.fill_between(x, y3 - y4, y3 + y4, color = 'red', alpha = 0.1)
    plt.fill_between(x, y5 - y6, y5 + y6, color = 'green', alpha = 0.1)
    
    plt.xlabel('Episodes', fontsize=12, labelpad=10)
    plt.ylabel('Average throughput (wafers/shift)', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    #plt.ylim(2900,3200)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(r"\batchsize_throughput.png", dpi=300, bbox_inches='tight') #Input the saving directory
    plt.show()

def draw_me(me1, me2, me3, window_size, sigma):
    ##DRAW THROUGHPUT (window_size = 50, sigma = 25)
    #Pre-process machine efficiency (scale up)
    for l in range(len(me1)):
        me1[l] = me1[l]*100
        me2[l] = me2[l]*100
        me3[l] = me3[l]*100
    
    #4. Calculate smooth_data and deviation
    smooth_me1 = smooth_gaussian(me1, sigma)
    smooth_me1_dev = compute_dev(smooth_me1, window_size)
    
    smooth_me2 = smooth_gaussian(me2, sigma)
    smooth_me2_dev = compute_dev(smooth_me2, window_size)
    
    smooth_me3 = smooth_gaussian(me3, sigma)
    smooth_me3_dev = compute_dev(smooth_me3, window_size)
    
    #5. Plot Backlog quantity
    #Define x-axis
    x = np.arange(eval_freq, train_episode + eval_freq, eval_freq)
    
    #Define mappo axis (y1, y2)
    y1 = smooth_me1
    y2 = smooth_me1_dev
    
    #Define ippo axis (y3, y4)
    y3 = smooth_me2
    y4 = smooth_me2_dev
    
    #Define ippo axis (y3, y4)
    y5 = smooth_me3
    y6 = smooth_me3_dev
    
    #plt.figure(figsize=(12,10))
    plt.plot(x, y1, label = 'b84', color='blue', alpha = 1)
    plt.plot(x, y3, label = 'b105', color='red', alpha = 1)
    plt.plot(x, y5, label = 'b140', color='green', alpha = 1)
    
    plt.fill_between(x, y1 - y2, y1 + y2, color = 'blue', alpha = 0.1)
    plt.fill_between(x, y3 - y4, y3 + y4, color = 'red', alpha = 0.1)
    plt.fill_between(x, y5 - y6, y5 + y6, color = 'green', alpha = 0.1)
    
    plt.xlabel('Episodes', fontsize=12, labelpad=10)
    plt.ylabel('Machine efficiency (%)', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    #plt.ylim(2900,3200)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(r"\batchsize_me.png", dpi=300, bbox_inches='tight') #Input the saving directory
    plt.show()

def draw_reward(reward1, reward2, reward3):
    ##DRAW REWARD
    #5. Plot Backlog quantity
    #Define x-axis
    x = np.arange(eval_freq, train_episode + eval_freq, eval_freq)
    
    #Define mappo axis (y1, y2)
    y1 = reward1
    y2 = reward2
    y3 = reward3
    
    #plt.figure(figsize=(8,6))
    plt.plot(x, y1, label = 'b84', color='blue', linewidth=2, alpha = 1)
    plt.plot(x, y2, label = 'b105', color='red', linewidth=2, alpha = 1)
    plt.plot(x, y3, label = 'b140', color='green', linewidth=2, alpha = 1)    
    
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Episode rewards', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 
    ax = plt.gca()  # Get the current axis
    #ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.legend()
    plt.savefig(r"\batchsize_reward.png", dpi=300, bbox_inches='tight') #Input the saving directory
    plt.show()

#Function to plot results
#draw_bl(b84_bl, b105_bl, b140_bl, window_size=60, sigma=10)
#draw_th(b84_th, b105_th, b140_th, window_size=60, sigma=10)
#draw_reward(b84_reward, b105_reward, b140_reward)
#draw_me(b84_me, b105_me, b140_me, window_size=60, sigma=10)

##----------------------------------------------------------------------------------------------------------------------------------

#1. Load result file using pickle
#1.2 Load MAPPO results
discrete_reward_dir = r'C:\Python\SFL\trained_MAPPO\Discrete_action\results\reward.pkl'
with open(discrete_reward_dir, 'rb') as file_13:
    discrete_reward = pickle.load(file_13)

discrete_th_dir = r'C:\Python\SFL\trained_MAPPO\Discrete_action\results\th.pkl'
with open(discrete_th_dir, 'rb') as file_14:
    discrete_th = pickle.load(file_14)
    
discrete_bl_dir = r'C:\Python\SFL\trained_MAPPO\Discrete_action\results\bl.pkl'
with open(discrete_bl_dir, 'rb') as file_15:
    discrete_bl = pickle.load(file_15)
    
discrete_me_dir = r'C:\Python\SFL\trained_MAPPO\Discrete_action\results\me.pkl'
with open(discrete_me_dir, 'rb') as file_16:
    discrete_me = pickle.load(file_16)
    
##----------------------------------------------------------------------------------------------------------------------------------
    
#1.3 Load IPPO results
multi_reward_dir = r'C:\Python\SFL\trained_MAPPO\B140\results\reward.pkl'
with open(multi_reward_dir, 'rb') as file_17:
    multi_reward = pickle.load(file_17)

multi_th_dir = r'C:\Python\SFL\trained_MAPPO\B140\results\th.pkl'
with open(multi_th_dir, 'rb') as file_18:
    multi_th = pickle.load(file_18)
    
multi_bl_dir = r'C:\Python\SFL\trained_MAPPO\B140\results\bl.pkl'
with open(multi_bl_dir, 'rb') as file_19:
    multi_bl = pickle.load(file_19)

multi_me_dir = r'C:\Python\SFL\trained_MAPPO\B140\results\me.pkl'
with open(multi_me_dir, 'rb') as file_20:
    multi_me = pickle.load(file_20)

##----------------------------------------------------------------------------------------------------------------------------------

#2. Define parameters
window_size = 60 #80
eval_freq = 20
train_episode = 5000
alpha = 2/(window_size+1)
sigma = 15 #30

#3. Define function for smoothing and computing deviation
def smooth_gaussian(data, sigma):
    smooth_data = copy.deepcopy(data)
    
    #Gaussian 
    smooth_data = gaussian_filter(smooth_data, sigma)
    
    return smooth_data
    
def compute_dev(smooth_data, window_size):
    half_window = window_size // 2
    deviation = np.zeros_like(smooth_data)
        
    for i in range(len(smooth_data)):
        start = max(0, i - half_window)
        end = min(len(smooth_data), i + half_window + 1)
        window_data = smooth_data[start:end]
        deviation[i] = np.std(window_data)
    return deviation

##DRAW BACKLOG (window_size = 30, sigma = 15)
#4. Calculate smooth_data and deviation
def draw_bl(bl1, bl2, window_size, sigma):
    smooth_bl1 = smooth_gaussian(bl1, sigma)
    smooth_bl1_dev = compute_dev(smooth_bl1, window_size)
    
    smooth_bl2 = smooth_gaussian(bl2, sigma)
    smooth_bl2_dev = compute_dev(smooth_bl2, window_size)
        
    #5. Plot Backlog quantity
    #Define x-axis
    x = np.arange(eval_freq, train_episode + eval_freq, eval_freq)
    
    #Define mappo axis (y1, y2)
    y1 = smooth_bl1
    y2 = smooth_bl1_dev
    
    #Define mappo axis (y3, y4)
    y3 = smooth_bl2[:250]
    y4 = smooth_bl2_dev[:250]
    
    
    #plt.figure(figsize=(3,6))
    plt.plot(x, y1, label = 'Multi-discrete', color='blue', alpha = 1)
    plt.plot(x, y3, label = 'Discrete', color='red', alpha = 1)
    
    plt.fill_between(x, y1 - y2, y1 + y2, color = 'blue', alpha = 0.1)
    plt.fill_between(x, y3 - y4, y3 + y4, color = 'red', alpha = 0.1)
    
    plt.xlabel('Episodes', fontsize=12, labelpad=10)
    plt.ylabel('Backlog quantity (wafers)', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    #plt.ylim(4000,9000)
    plt.legend()
    plt.grid(True)
    plt.savefig(r"\action_space_backlog.png", dpi=300, bbox_inches='tight') #Input the saving directory
    plt.show()


def draw_th(th1, th2, window_size, sigma):
    ##DRAW THROUGHPUT (window_size = 50, sigma = 25)
    #4. Calculate smooth_data and deviation
    smooth_th1 = smooth_gaussian(th1, sigma)
    smooth_th1_dev = compute_dev(smooth_th1, window_size)
    
    smooth_th2 = smooth_gaussian(th2, sigma)
    smooth_th2_dev = compute_dev(smooth_th2, window_size)
    
    #5. Plot Backlog quantity
    #Define x-axis
    x = np.arange(eval_freq, train_episode + eval_freq, eval_freq)
    
    #Define mappo axis (y1, y2)
    y1 = smooth_th1
    y2 = smooth_th1_dev
    
    #Define ippo axis (y3, y4)
    y3 = smooth_th2[:250]
    y4 = smooth_th2_dev[:250]
        
    #plt.figure(figsize=(12,10))
    plt.plot(x, y1, label = 'Multi-discrete', color='blue', alpha = 1)
    plt.plot(x, y3, label = 'Discrete', color='red', alpha = 1)
    
    plt.fill_between(x, y1 - y2, y1 + y2, color = 'blue', alpha = 0.1)
    plt.fill_between(x, y3 - y4, y3 + y4, color = 'red', alpha = 0.1)
    
    plt.xlabel('Episodes', fontsize=12, labelpad=10)
    plt.ylabel('Average throughput (wafers/shift)', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    #plt.ylim(2900,3200)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(r"\action_space_throughput.png", dpi=300, bbox_inches='tight') #Input the saving directory
    plt.show()

def draw_me(me1, me2, window_size, sigma):
    ##DRAW THROUGHPUT (window_size = 50, sigma = 25)
    #Pre-process machine efficiency (scale up)
    for l in range(len(me1)):
        me1[l] = me1[l]*100
        me2[l] = me2[l]*100
        
    #4. Calculate smooth_data and deviation
    smooth_me1 = smooth_gaussian(me1, sigma)
    smooth_me1_dev = compute_dev(smooth_me1, window_size)
    
    smooth_me2 = smooth_gaussian(me2, sigma)
    smooth_me2_dev = compute_dev(smooth_me2, window_size)
    
    #5. Plot Backlog quantity
    #Define x-axis
    x = np.arange(eval_freq, train_episode + eval_freq, eval_freq)
    
    #Define mappo axis (y1, y2)
    y1 = smooth_me1
    y2 = smooth_me1_dev
    
    #Define ippo axis (y3, y4)
    y3 = smooth_me2[:250]
    y4 = smooth_me2_dev[:250]
    
    #Define optimal result
    #y5 = th3
    
    #plt.figure(figsize=(12,10))
    plt.plot(x, y1, label = 'Multi-discrete', color='blue', alpha = 1)
    plt.plot(x, y3, label = 'Discrete', color='red', alpha = 1)
    #plt.plot(x, y5, label = 'Optimal', color='black', alpha = 1, linestyle='--')
    
    plt.fill_between(x, y1 - y2, y1 + y2, color = 'blue', alpha = 0.1)
    plt.fill_between(x, y3 - y4, y3 + y4, color = 'red', alpha = 0.1)
    
    plt.xlabel('Episodes', fontsize=12, labelpad=10)
    plt.ylabel('Machine efficiency (%)', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    #plt.ylim(2900,3200)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(r"\action_space_me.png", dpi=300, bbox_inches='tight') #Input the saving directory
    plt.show()


def draw_reward(reward1, reward2):
    ##DRAW REWARD
    #5. Plot Backlog quantity
    #Define x-axis
    x = np.arange(eval_freq, train_episode + eval_freq, eval_freq)
    
    #Define mappo axis (y1, y2)
    y1 = reward1
    y2 = reward2[:250]
    
    #plt.figure(figsize=(8,6))
    plt.plot(x, y1, label = 'Multi-discrete', color='blue', linewidth=2, alpha = 1)
    plt.plot(x, y2, label = 'Discrete', color='red', linewidth=2, alpha = 1)
    
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Episode rewards', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.legend()
    plt.savefig(r"\action_space_reward.png", dpi=300, bbox_inches='tight') #Input the saving directory
    plt.show()


#Function to plot results
#draw_bl(multi_bl, discrete_bl, window_size=60, sigma=10)
#draw_th(multi_th, discrete_th, window_size=60, sigma=10)
#draw_reward(multi_reward, discrete_reward)
#draw_me(multi_me, discrete_me, window_size=60, sigma=10)

##----------------------------------------------------------------------------------------------------------------------------------

#1. Load result file using pickle
#1.2 Load MAPPO results
MAPPO_reward_dir = r'C:\Python\SFL\trained_MAPPO\B140\results\reward.pkl'
with open(MAPPO_reward_dir, 'rb') as file_21:
    MAPPO_reward = pickle.load(file_21)

MAPPO_th_dir = r'C:\Python\SFL\trained_MAPPO\B140\results\th.pkl'
with open(MAPPO_th_dir, 'rb') as file_22:
    MAPPO_th = pickle.load(file_22)
    
MAPPO_bl_dir = r'C:\Python\SFL\trained_MAPPO\B140\results\bl.pkl'
with open(MAPPO_bl_dir, 'rb') as file_23:
    MAPPO_bl = pickle.load(file_23)

MAPPO_me_dir = r'C:\Python\SFL\trained_MAPPO\B140\results\me.pkl'
with open(MAPPO_me_dir, 'rb') as file_24:
    MAPPO_me = pickle.load(file_24)    

##----------------------------------------------------------------------------------------------------------------------------------    
 
#1.3 Load IPPO results
IPPO_reward_dir = r'C:\Python\SFL\trained_IPPO\2P\results\reward.pkl'
with open(IPPO_reward_dir, 'rb') as file_25:
    IPPO_reward = pickle.load(file_25)

IPPO_th_dir = r'C:\Python\SFL\trained_IPPO\2P\results\th.pkl'
with open(IPPO_th_dir, 'rb') as file_26:
    IPPO_th = pickle.load(file_26)
    
IPPO_bl_dir = r'C:\Python\SFL\trained_IPPO\2P\results\bl.pkl'
with open(IPPO_bl_dir, 'rb') as file_27:
    IPPO_bl = pickle.load(file_27)
    
IPPO_me_dir = r'C:\Python\SFL\trained_IPPO\2P\results\me.pkl'
with open(IPPO_me_dir, 'rb') as file_28:
    IPPO_me = pickle.load(file_28)

##----------------------------------------------------------------------------------------------------------------------------------    

#2. Define parameters
window_size = 60 #80
eval_freq = 20
train_episode = 5000
alpha = 2/(window_size+1)
sigma = 15 #30

#3. Define function for smoothing and computing deviation
def smooth_gaussian(data, sigma):
    smooth_data = copy.deepcopy(data)
    
    #Gaussian 
    smooth_data = gaussian_filter(smooth_data, sigma)
    
    return smooth_data
    
def compute_dev(smooth_data, window_size):
    half_window = window_size // 2
    deviation = np.zeros_like(smooth_data)
        
    for i in range(len(smooth_data)):
        start = max(0, i - half_window)
        end = min(len(smooth_data), i + half_window + 1)
        window_data = smooth_data[start:end]
        deviation[i] = np.std(window_data)
    return deviation

##DRAW BACKLOG (window_size = 30, sigma = 15)
#4. Calculate smooth_data and deviation
def draw_bl(bl1, bl2, window_size, sigma):
    smooth_bl1 = smooth_gaussian(bl1, sigma)
    smooth_bl1_dev = compute_dev(smooth_bl1, window_size)
    
    smooth_bl2 = smooth_gaussian(bl2, sigma)
    smooth_bl2_dev = compute_dev(smooth_bl2, window_size)
        
    #5. Plot Backlog quantity
    #Define x-axis
    x = np.arange(eval_freq, train_episode + eval_freq, eval_freq)
    
    #Define mappo axis (y1, y2)
    y1 = smooth_bl1
    y2 = smooth_bl1_dev
    
    #Define mappo axis (y3, y4)
    y3 = smooth_bl2[:250]
    y4 = smooth_bl2_dev[:250]
    
    #Define optimal result
    
    #plt.figure(figsize=(3,6))
    plt.plot(x, y1, label = 'MAPPO', color='blue', alpha = 1)
    plt.plot(x, y3, label = 'IPPO', color='red', alpha = 1)    
    
    plt.fill_between(x, y1 - y2, y1 + y2, color = 'blue', alpha = 0.1)
    plt.fill_between(x, y3 - y4, y3 + y4, color = 'red', alpha = 0.1)
    
    plt.xlabel('Episodes', fontsize=12, labelpad=10)
    plt.ylabel('Backlog quantity (wafers)', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    #plt.ylim(4000,9000)
    plt.legend(loc='center right')
    plt.grid(True)
    plt.savefig(r"C:\Users\user\Desktop\논문심사자료\국제저널용\High reso figure\RL comparison\RL_backlog.png", dpi=300, bbox_inches='tight') #Input the saving directory
    plt.show()

def draw_th(th1, th2, window_size, sigma):
    ##DRAW THROUGHPUT (window_size = 50, sigma = 25)
    #4. Calculate smooth_data and deviation
    smooth_th1 = smooth_gaussian(th1, sigma)
    smooth_th1_dev = compute_dev(smooth_th1, window_size)
    
    smooth_th2 = smooth_gaussian(th2, sigma)
    smooth_th2_dev = compute_dev(smooth_th2, window_size)
    
    #5. Plot Backlog quantity
    #Define x-axis
    x = np.arange(eval_freq, train_episode + eval_freq, eval_freq)
    
    #Define mappo axis (y1, y2)
    y1 = smooth_th1
    y2 = smooth_th1_dev
    
    #Define ippo axis (y3, y4)
    y3 = smooth_th2[:250]
    y4 = smooth_th2_dev[:250]
    
    #plt.figure(figsize=(12,10))
    plt.plot(x, y1, label = 'MAPPO', color='blue', alpha = 1)
    plt.plot(x, y3, label = 'IPPO', color='red', alpha = 1)
    
    plt.fill_between(x, y1 - y2, y1 + y2, color = 'blue', alpha = 0.1)
    plt.fill_between(x, y3 - y4, y3 + y4, color = 'red', alpha = 0.1)
    
    plt.xlabel('Episodes', fontsize=12, labelpad=10)
    plt.ylabel('Average throughput (wafers/shift)', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    #plt.ylim(2900,3200)
    plt.legend(loc='center right')
    plt.grid(True)
    plt.savefig(r"C:\Users\user\Desktop\논문심사자료\국제저널용\High reso figure\RL comparison\RL_throughput.png", dpi=300, bbox_inches='tight') #Input the saving directory
    plt.show()
    
def draw_me(me1, me2, window_size, sigma):
    ##DRAW THROUGHPUT (window_size = 50, sigma = 25)
    #Pre-process machine efficiency (scale up)
    for l in range(len(me1)):
        me1[l] = me1[l]*100
        me2[l] = me2[l]*100
        
    #4. Calculate smooth_data and deviation
    smooth_me1 = smooth_gaussian(me1, sigma)
    smooth_me1_dev = compute_dev(smooth_me1, window_size)
    
    smooth_me2 = smooth_gaussian(me2, sigma)
    smooth_me2_dev = compute_dev(smooth_me2, window_size)
    
    #5. Plot Backlog quantity
    #Define x-axis
    x = np.arange(eval_freq, train_episode + eval_freq, eval_freq)
    
    #Define mappo axis (y1, y2)
    y1 = smooth_me1
    y2 = smooth_me1_dev
    
    #Define ippo axis (y3, y4)
    y3 = smooth_me2[:250]
    y4 = smooth_me2_dev[:250]
    
    
    #plt.figure(figsize=(12,10))
    plt.plot(x, y1, label = 'MAPPO', color='blue', alpha = 1)
    plt.plot(x, y3, label = 'IPPO', color='red', alpha = 1)
    
    plt.fill_between(x, y1 - y2, y1 + y2, color = 'blue', alpha = 0.1)
    plt.fill_between(x, y3 - y4, y3 + y4, color = 'red', alpha = 0.1)
    
    plt.xlabel('Episodes', fontsize=12, labelpad=10)
    plt.ylabel('Machine efficiency (%)', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.ylim(84,91)
    plt.legend()
    plt.grid(True)
    plt.savefig(r"C:\Users\user\Desktop\논문심사자료\국제저널용\High reso figure\RL comparison\RL_me.png", dpi=300, bbox_inches='tight') #Input the saving directory
    plt.show()


def draw_reward(reward1, reward2):
    ##DRAW REWARD
    #5. Plot Backlog quantity
    #Define x-axis
    x = np.arange(eval_freq, train_episode + eval_freq, eval_freq)
    
    #Define mappo axis (y1, y2)
    y1 = reward1
    y2 = reward2[:250]
    
    #plt.figure(figsize=(8,6))
    plt.plot(x, y1, label = 'MAPPO', color='blue', linewidth=2, alpha = 1)
    plt.plot(x, y2, label = 'IPPO', color='red', linewidth=2, alpha = 1)
    
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Episode rewards', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10) 
    ax = plt.gca()  # Get the current axis
    #ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.legend(loc='lower right')
    plt.savefig(r"C:\Users\user\Desktop\논문심사자료\국제저널용\High reso figure\RL comparison\RL_reward.png", dpi=300, bbox_inches='tight') #Input the saving directory
    plt.show()


#Function to plot results
draw_bl(MAPPO_bl, IPPO_bl, window_size=60, sigma=10)
draw_th(MAPPO_th, IPPO_th, window_size=60, sigma=10)
draw_reward(MAPPO_reward, IPPO_reward,)
#draw_me(MAPPO_me, IPPO_me, window_size=60, sigma=10)











