import numpy as np
import pickle, os, torch
from torch.nn import functional as F
from code_repository import test_LR, Log_Reg, My_plot
import matplotlib.pyplot as plt


import os
os.chdir('.\\Satellite')


dataset = 'Bank' # Satellite, Bank, Robot, Synthetic_data
# LR_Model_syn_NT.pckl
# LR_Model_Bank_NT.pckl
# LR_Model_Robot_NT.pckl
# LR_Model_Satellite_NT.pckl
filename = 'LR_Model_Satellite_NT.pckl'
# Synthetic_data



#To see what we extract in Main_Model_Params see 1-Main.py
f = open(filename, 'rb')
Main_Model_Params = pickle.load(f)
f.close()


Main_X = torch.tensor(Main_Model_Params[1])
Main_pred = F.softmax(Main_Model_Params[0](Main_X), dim=1).detach().numpy()
dt, k0, i = Main_X.shape[1], 0, 0
t0 = int(0.9*dt)

PPS_res = []
while k0<dt:
    i = 0
    while i<t0:
        
        PPS_name = 'Model_PPS3_'+str(k0)+'_'+str(i)+'.pckl'
        f = open(PPS_name, 'rb')
        PPS_model = pickle.load(f)
        f.close()
        PPS_X = torch.tensor(PPS_model[1])
        PPS_pred = F.softmax(PPS_model[0](PPS_X), dim=1).detach().numpy()
        
        PPS_res.append( [k0, i, np.sum(np.abs(Main_pred-PPS_pred))/(2*Main_pred.shape[0])] )
        
        i+=1
    k0+=1
    print(k0)
    
f = open(dataset+'_TV_PPS3.pckl', 'wb') # TV stands for total variation 
pickle.dump(PPS_res, f)
f.close()

t1, PPS_TV = My_plot(dt, dataset+'_TV_PPS3.pckl')

plt.plot(t1, PPS_TV, '-ob')
# plt.legend(['Total variation PPS3'])
plt.ylabel('Average total variation distance', fontsize = 15)
plt.xlabel('Ratio of passive party features', fontsize = 15)
plt.grid()
plt.show()



# tensor([[ 0.4290, -4.8065,  0.5092,  0.4345,  0.4788],
#         [-0.4290,  4.8065, -0.5092, -0.4345, -0.4788]],

# W = list(Main_Model_Params[0].named_parameters())[0][1]

# PPS_name = 'Model_PPS3_'+str(0)+'_'+str(1)+'.pckl'
# f = open(PPS_name, 'rb')
# PPS_model = pickle.load(f)
# f.close()

# W1 = list(PPS_model[0].named_parameters())[0][1]

# print(W, W1)
        
        
        
        


