
import pickle
import numpy as np
import matplotlib.pyplot as plt
from LT_codes import My_plot
import itertools
marker = itertools.cycle(('o',',', '+', '.',  '*')) 
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
#1: Drive 2: Satellite

%matplotlib qt
d_tot = 5

fig, axs = plt.subplots(2,3)

LOSS = ['ESA_NoClamp','ESA_Clamp','ESA_Clamp_Extended', 'ESA_0centre', 'ESA_RCC', 'ESA_CLS']
LOSS1 = [('LS', 'LS PPS1, LS PPS2'),('Clamped LS', 'Clamped LS PPS1', 'Clamped LS PPS2'),
         ('Half*', 'Half* PPS1', 'Half* PPS2'),('RCC1', 'RCC1 PPS1', 'RCC1 PPS2'),('RCC2', 'RCC2 PPS1', 'RCC2 PPS2'),('CLS', 'CLS PPS1', 'CLS PPS2'),]
C = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]
for i in range(6):
    STR = 'Bef_Trans_'
    filename = STR+LOSS[i]+'.pckl'
    t1, MSE_plt = My_plot(d_tot, filename)
    axs[C[i][0]][C[i][1]].plot(t1, MSE_plt, "-", marker = 'o', label=i)
    
    STR = 'PPS1_Trans_'
    filename = STR+LOSS[i]+'.pckl'
    t1, MSE_plt = My_plot(d_tot, filename)
    axs[C[i][0]][C[i][1]].plot(t1, MSE_plt, ".-", marker = '*', label=i)
    
    STR = 'Aft_Trans_'
    filename = STR+LOSS[i]+'.pckl'
    t1, MSE_plt = My_plot(d_tot, filename)
    axs[C[i][0]][C[i][1]].plot(t1, MSE_plt, "--", marker = 'x', label=i)
    
    axs[C[i][0]][C[i][1]].legend(LOSS1[i], loc='lower right')
    axs[C[i][0]][C[i][1]].set_ylabel('MSE', fontsize=15)
    axs[C[i][0]][C[i][1]].set_xlabel('Ratio of passive party features', fontsize=15)
    axs[C[i][0]][C[i][1]].grid()


    





    
 

    
    
    
    
# f = open('Satellite_Bor1_NoClamp_'+str(3)+'.pckl', 'rb')#########
# MSE = np.array(pickle.load(f))
# f.close()
# for i in range(2,4):##########
#     f = open('Satellite_KLD1_NoClamp_'+str(i)+'.pckl', 'rb')
#     MSE[:,2] += np.array(pickle.load(f))[:,2]
#     f.close()
# MSE[:,2]/=3######

            
# f = open('Satellite_KLD1_NoClamp.pckl', 'wb')
# pickle.dump(MSE, f)
# f.close()
