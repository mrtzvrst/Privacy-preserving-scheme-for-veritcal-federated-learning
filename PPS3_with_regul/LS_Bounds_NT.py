import os, itertools
import numpy as np
import torch, pickle
import scipy as sc
os.chdir('..')
from code_repository import My_plot
import matplotlib.pyplot as plt
from os.path import exists

torch.set_default_dtype(torch.float64)
%matplotlib qt

"""Extracting the pre-trained model"""
LOSS = 'ESA'
#'Satellite'
#'Bank'
#'Robot'
#'Synthetic'
STR = 'Satellite'
dt = 36
os.chdir(os.getcwd()+'\\PPS3_with_regul\\'+STR)
#'LR_Model_Synthetic_NT.pckl'
#'LR_Model_Bank_NT.pckl'
#'LR_model_robot_NT.pckl'
#'LR_model_Satellite_NT.pckl'
f = open('LR_model_Satellite_NT.pckl', 'rb')
model_data = pickle.load(f)
f.close()



def Bounds(Weights, X, STR):
    Num_of_Features = Weights.shape[1]
       
    t0 = int(0.9*Num_of_Features)
    LS, Half_star = np.zeros((1, t0)), np.zeros((1, t0))
    k0 = 0
    while k0 < Num_of_Features:
        print(k0)
        i = 0
        while i < t0:
            missing_features = np.mod(np.arange(k0,i+1+k0), Num_of_Features)#np.arange(47,47-i-1,-1) #np.arange(12,i+1+12)  #sample(Feature_index, i+1) #np.arange(0,i+1) 
            
            dpas = len(missing_features)
            Xn = X[:, missing_features]
            Half = np.reshape(np.ones((1, dpas))/2, (-1,1))
            KZero = np.zeros((dpas, dpas))
            KHalf = np.zeros((dpas, dpas))
            for j in range(Xn.shape[0]):
                temp = np.reshape(Xn[j, :], (-1,1))
                KZero += np.matmul(temp, temp.T)
                KHalf += np.matmul(temp-Half, (temp-Half).T)
                
            KZero/=Xn.shape[0]
            KHalf/=Xn.shape[0]
            
            
            Wpas = Weights[:, missing_features]
            A = Wpas[0:-1,:]-Wpas[1:,:]
            Aplus = np.linalg.pinv(A)
            
            if dpas!=1:
                Null_A = sc.linalg.null_space(A).shape[1]
                LS[0,i] += np.trace( np.matmul( (np.identity(dpas)-np.matmul(Aplus, A)) , KZero) )/dpas
                Half_star[0,i] += np.trace( np.matmul( (np.identity(dpas)-np.matmul(Aplus, A)) , KHalf) )/dpas
            i+=1
        k0+=1
    
    f = open(STR+'LS', 'wb')
    pickle.dump(LS/Num_of_Features, f)
    f.close()
    
    f = open(STR+'Half_star', 'wb')
    pickle.dump(Half_star/Num_of_Features, f)
    f.close()
    

if not exists(STR+'_Half_star') and not exists(STR+'_LS'):
    Weights = np.array(list(model_data[0].named_parameters())[0][1].detach().cpu(), dtype = np.float64)
    X = model_data[3] # This is X_train 
    Bounds(Weights, X, STR+'_')


# marker = itertools.cycle(('o',',', '+', '.',  '*')) 

t1, LS_emp = My_plot(dt, 'Bef_Trans_ESA_NoClamp.pckl')
t2, Half_emp = My_plot(dt, 'Bef_Trans_ESA_Clamp_Extended.pckl')

f = open(STR+'_LS', 'rb')
LS = pickle.load(f)
f.close()

f = open(STR+'_Half_star', 'rb')
Half_star = pickle.load(f)
f.close()

plt.plot(t1, LS_emp, 'o')
plt.plot(t1, Half_emp, '+')
plt.plot(t1, LS.flatten(), '-,')
plt.plot(t1, Half_star.flatten(), '-*')
plt.show()

