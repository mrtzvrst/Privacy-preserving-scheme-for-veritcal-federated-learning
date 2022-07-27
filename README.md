# Privacy-preserving-scheme-for-veritcal-federated-learning

These are the two schemes proposed in our paper https://arxiv.org/abs/2207.11788

All the required codes are available in code_repository.py, however, this is not directly used.
One needs only to run 1-Main.py. To that end simply put the required dateset in the same directory as 1-Main.py and code_repository.py and run 1-Main.py (you would need to make sure the variables for choosing the preferred dataset are set in 1-Main.py). 

The results have been obtained for 4 datasets (Bank, Robot, Satellite and Synthetic data) and put in the relevant folder. To see the relevant plots, copy 1-plot and code_repository.py in the same folder and run 1-plot.py.

If you want to add a new dataset, you would need to add the way you read the data in "Read_data" function in code_repository.py and write similar codes in 1_Main.py. 
