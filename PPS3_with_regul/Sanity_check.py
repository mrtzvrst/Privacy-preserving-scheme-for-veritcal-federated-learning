import torch , os
import matplotlib.pyplot as plt
os.chdir('..')
from code_repository import My_plot, Diff_Bound
os.chdir('.\\PPS3_with_regul')

# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
import pickle


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.set_default_dtype(torch.float64)




dataset = 'Robot'
dt = 24

os.chdir('./'+dataset)


t1, LS = My_plot(dt, 'Bef_Trans_ESA_NoClamp.pckl')
t1, LS_PPS = My_plot(dt, 'PPS3_Trans_ESA_NoClamp.pckl')
Diff_LS = LS_PPS- LS

t1, Half_st = My_plot(dt, 'Bef_Trans_ESA_Clamp_Extended.pckl')
t1, Half_st_PPS = My_plot(dt, 'PPS3_Trans_ESA_Clamp_Extended.pckl')
Diff_Half_st = Half_st_PPS- Half_st

f = open('LR_Model_'+dataset+'_NT.pckl', 'rb')
Model = pickle.load(f)
f.close()
Weights = list(Model[0].parameters())[0].detach().numpy()
Diff_Th = Diff_Bound(Weights, Model[3])


plt.plot(Diff_LS, 'b')
plt.plot(Diff_Half_st, 'r')
plt.plot(Diff_Th.flatten(), 'k')
plt.show()