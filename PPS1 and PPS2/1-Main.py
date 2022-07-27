import torch 
import numpy as np
from torch import nn, optim
from code_repository import random_mini_batches, Log_Reg, Read_data, test_LR, train_LR, ESA, PPS1, PPS2
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
import pickle
from os.path import exists

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.set_default_dtype(torch.float64)

Read = 2 #Bank:2, Satellite:3, Robot:4, Synthetic:5


Lambda, learning_rate, seed = 0, 0.05, 0
k0_stp, i_stp, Num_of_Predictions  = 1, 1, 200

if Read==2:
    accuracy = 0.9
    Model_NT_name = 'LR_Model_Bank_NT.pckl'
    input_dim, output_dim, epochs, batch_size, seed = 19, 2, 600, 1000, 0
    if not exists(Model_NT_name):
        X, Y= Read_data('bank-additional-full.csv')
        """Normalization"""
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        """Train/test/validation set"""
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=40)
        X_test, X_valid, Y_test, Y_valid = train_test_split(X_test,Y_test, test_size = 0.5, random_state=40)
        
        X_train, X_test, X_valid = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(X_valid)
        Y_train, Y_test, Y_valid = torch.LongTensor(Y_train), torch.LongTensor(Y_test), torch.LongTensor(Y_valid)
                
        #Training the data
        model_NT = Log_Reg(input_dim, output_dim) # Model with No Transformation
        CEF_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_NT.parameters(), lr=learning_rate)
        Final_model_NT = train_LR(model_NT, epochs, optimizer, CEF_loss, Lambda, batch_size, X_train, Y_train, X_valid, Y_valid, seed)
        _ = test_LR(Final_model_NT, X_test, Y_test, X_train, Y_train)
        
        #Saving the model and params
        PARAM = (Final_model_NT, X, Y, X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy(), X_valid.numpy(), Y_valid.numpy(), Lambda, learning_rate)
        
        f = open(Model_NT_name, 'wb')
        pickle.dump(PARAM, f)
        f.close()
    
    
elif Read==3:
    accuracy = 0.88
    Model_NT_name = 'LR_Model_Satellite_NT.pckl'
    input_dim, output_dim, epochs, batch_size, seed = 36, 6, 600, 1000, 0
    if not exists(Model_NT_name):
        X_train, X_test, Y_train, Y_test = Read_data('Sat_train.txt')
        # """Outlier detection"""
        # X, Y = Outlier_detection(X, Y, contam_factor=0.05)
        """Normalization"""
        X_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))
        X_test = (X_test - np.min(X_test, axis=0)) / (np.max(X_test, axis=0) - np.min(X_test, axis=0))
    
        X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
        Y_train, Y_test = torch.LongTensor(Y_train), torch.LongTensor(Y_test)
        X_valid, Y_valid = X_test, Y_test
    
        #Training the data
        model_NT = Log_Reg(input_dim, output_dim) # Model with No Transformation
        CEF_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_NT.parameters(), lr=learning_rate)
        Final_model_NT = train_LR(model_NT, epochs, optimizer, CEF_loss, Lambda, batch_size, X_train, Y_train, X_valid, Y_valid, seed)
        _ = test_LR(Final_model_NT, X_test, Y_test, X_train, Y_train)
        
        #Saving the model and params
        PARAM = (Final_model_NT, torch.cat((X_train, X_test), dim=0).numpy(), torch.cat((Y_train, Y_test), dim=0).numpy(), X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy(), X_valid.numpy(), Y_valid.numpy(), Lambda, learning_rate)
        
        f = open(Model_NT_name, 'wb')
        pickle.dump(PARAM, f)
        f.close()
    
elif Read==4:
    accuracy = 0.69
    Model_NT_name = 'LR_Model_Robot_NT.pckl'
    input_dim, output_dim, epochs, batch_size, seed = 24, 4, 500, 1000, 0
    if not exists(Model_NT_name):
        X, Y = Read_data('sensor_readings_24_data.txt')
        """Normalization"""
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        """Train/test/validation set"""
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=40)
        X_test, X_valid, Y_test, Y_valid = train_test_split(X_test,Y_test, test_size = 0.5, random_state=40)
        
        X_train, X_test, X_valid = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(X_valid)
        Y_train, Y_test, Y_valid = torch.LongTensor(Y_train), torch.LongTensor(Y_test), torch.LongTensor(Y_valid)
                
        #Training the data
        model_NT = Log_Reg(input_dim, output_dim) # Model with No Transformation
        CEF_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_NT.parameters(), lr=learning_rate)
        Final_model_NT = train_LR(model_NT, epochs, optimizer, CEF_loss, Lambda, batch_size, X_train, Y_train, X_valid, Y_valid, seed)
        _ = test_LR(Final_model_NT, X_test, Y_test, X_train, Y_train)
        
        #Saving the model and params
        PARAM = (Final_model_NT, X, Y, X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy(), X_valid.numpy(), Y_valid.numpy(), Lambda, learning_rate)
        
        f = open(Model_NT_name, 'wb')
        pickle.dump(PARAM, f)
        f.close()
    

elif Read==5:     
    accuracy = 0.94
    """First we train the data without any transformation"""
    Model_NT_name = 'LR_Model_syn_NT.pckl'    
    input_dim, output_dim, epochs, batch_size, seed = 5, 2, 400, 1000, 0
    if not exists(Model_NT_name):
        X,Y = make_classification(n_samples=50000, n_features=input_dim, n_informative=1, n_redundant=0, n_repeated=0, 
                                  n_classes=output_dim, n_clusters_per_class=1, weights=None, 
                                  flip_y=0.1, class_sep=1.0, hypercube=True, 
                                  shift=1.0, scale=1.0, shuffle=True, random_state=None)
        
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        
        #Train/test/validation set
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=40)
        X_test, X_valid, Y_test, Y_valid = train_test_split(X_test,Y_test, test_size = 0.5, random_state=40)
        
        X_train, X_test, X_valid = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(X_valid)
        Y_train, Y_test, Y_valid = torch.LongTensor(Y_train), torch.LongTensor(Y_test), torch.LongTensor(Y_valid)
        
        #Training the data
        model_NT = Log_Reg(input_dim, output_dim) # Model with No Transformation
        CEF_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_NT.parameters(), lr=learning_rate)
        Final_model_NT = train_LR(model_NT, epochs, optimizer, CEF_loss, Lambda, batch_size, X_train, Y_train, X_valid, Y_valid, seed)
        _ = test_LR(Final_model_NT, X_test, Y_test, X_train, Y_train)
        
        #Saving the model and params
        PARAM = (Final_model_NT, X, Y, X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy(), X_valid.numpy(), Y_valid.numpy(), Lambda, learning_rate)
        
        f = open(Model_NT_name, 'wb')
        pickle.dump(PARAM, f)
        f.close()


LOSS = 'ESA'

"""Results for the model without Linear transformation"""
STR = 'Bef_Trans_'
f = open(Model_NT_name, 'rb')
model_data = pickle.load(f)
f.close()

Weights = np.array(list(model_data[0].named_parameters())[0][1].detach().cpu(), dtype = np.float64)
Biases = np.array(list(model_data[0].named_parameters())[1][1].detach().cpu(), dtype = np.float64)
X, Y = model_data[1], model_data[2]

MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp.pckl', STR, 'yes', 'Non_extension')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp_Extended.pckl', STR, 'yes', 'Extended')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_NoClamp.pckl', STR, 'no', 'Non_extension')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_0centre.pckl', STR, 'no', '0centre')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_RCC.pckl', STR, 'no', 'RCC')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_CLS.pckl', STR, 'no', 'CLS')


"""Results for the model with transformation"""
STR = 'PPS1_Trans_'
MSE = PPS1(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp.pckl', STR, 'yes', 'Non_extension', epochs, batch_size, accuracy, Model_NT_name)
MSE = PPS1(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp_Extended.pckl', STR, 'yes', 'Extended', epochs, batch_size, accuracy, Model_NT_name)
MSE = PPS1(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_NoClamp.pckl', STR, 'no', 'Non_extension', epochs, batch_size, accuracy, Model_NT_name)
MSE = PPS1(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_0centre.pckl', STR, 'no', '0centre', epochs, batch_size, accuracy, Model_NT_name)
MSE = PPS1(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_RCC.pckl', STR, 'no', 'RCC', epochs, batch_size, accuracy, Model_NT_name)
MSE = PPS1(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_CLS.pckl', STR, 'no', 'CLS', epochs, batch_size, accuracy, Model_NT_name)


STR = 'PPS2_Trans_'
MSE = PPS2(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp.pckl', STR, 'yes', 'Non_extension', epochs, batch_size, accuracy, Model_NT_name)
MSE = PPS2(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp_Extended.pckl', STR, 'yes', 'Extended', epochs, batch_size, accuracy, Model_NT_name)
MSE = PPS2(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_NoClamp.pckl', STR, 'no', 'Non_extension', epochs, batch_size, accuracy, Model_NT_name)
MSE = PPS2(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_0centre.pckl', STR, 'no', '0centre', epochs, batch_size, accuracy, Model_NT_name)
MSE = PPS2(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_RCC.pckl', STR, 'no', 'RCC', epochs, batch_size, accuracy, Model_NT_name)
MSE = PPS2(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_CLS.pckl', STR, 'no', 'CLS', epochs, batch_size, accuracy, Model_NT_name)
