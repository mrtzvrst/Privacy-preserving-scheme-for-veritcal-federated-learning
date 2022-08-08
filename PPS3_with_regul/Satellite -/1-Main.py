import torch , os
import numpy as np
from torch import nn, optim


from code_repository import Log_Reg, Read_data, test_LR, train_LR_PPS, ESA, PPS3, Log_Reg_PPS


from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
import pickle
from os.path import exists

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.set_default_dtype(torch.float64)

Read = 'Satellite' # Bank, Satellite, Robot, Synthetic

learning_rate = 0.1
k0_stp, i_stp, Num_of_Predictions  = 1, 1, 200

if Read=='Bank':
    os.chdir('./Bank')
    
    accuracy, Lambda1, Lambda2 = 0.5, 0.001, 10**-13
    Model_NT_name = 'LR_Model_Bank_NT.pckl'
    input_dim, output_dim, epochs, batch_size = 19, 2, 600, 1000
    if not exists(Model_NT_name):
        X, Y= Read_data('bank-additional-full.csv')
        """Normalization"""
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        """Train/test/validation set"""
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)
        X_test, X_valid, Y_test, Y_valid = train_test_split(X_test,Y_test, test_size = 0.5, random_state=0)
        
        X_train, X_test, X_valid = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(X_valid)
        Y_train, Y_test, Y_valid = torch.LongTensor(Y_train), torch.LongTensor(Y_test), torch.LongTensor(Y_valid)
                
        #Training the data
        model_NT = Log_Reg(input_dim, output_dim) # Model with No Transformation
        CEF_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_NT.parameters(), lr=learning_rate)
        Final_model_NT = train_LR_PPS(model_NT, epochs, optimizer, CEF_loss, Lambda1, Lambda2, batch_size, 
                                  X_train, Y_train, X_valid, Y_valid, 0)
        _ = test_LR(Final_model_NT, X_test, Y_test, X_train, Y_train)
        
        #Saving the model and params
        PARAM = (Final_model_NT, X, Y, X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy(), X_valid.numpy(), Y_valid.numpy(), Lambda1, Lambda2, learning_rate)
        
        f = open(Model_NT_name, 'wb')
        pickle.dump(PARAM, f)
        f.close()
    
    
elif Read=='Satellite':
    
    
    accuracy, Lambda1, Lambda2 = 0.5, 0.0002, 10**-20
    Model_NT_name = 'LR_Model_Satellite_NT.pckl'
    input_dim, output_dim, epochs, batch_size = 36, 6, 3000, 1000
    if not exists(Model_NT_name):
        X_tr, X_ts, Y_tr, Y_ts = Read_data('Sat_train.txt')
        X, Y = np.concatenate((X_tr, X_ts), axis = 0), np.concatenate((Y_tr, Y_ts), axis=0)
        """Normalization"""
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        """Train/test/validation set"""
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)
        X_test, X_valid, Y_test, Y_valid = train_test_split(X_test,Y_test, test_size = 0.5, random_state=0)
        
        X_train, X_test, X_valid = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(X_valid)
        Y_train, Y_test, Y_valid = torch.LongTensor(Y_train), torch.LongTensor(Y_test), torch.LongTensor(Y_valid)
    
        #Training the data
        model_NT = Log_Reg(input_dim, output_dim) # Model with No Transformation
        CEF_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_NT.parameters(), lr=learning_rate)
        Final_model_NT = train_LR_PPS(model_NT, epochs, optimizer, CEF_loss, Lambda1, Lambda2, batch_size, 
                                  X_train, Y_train, X_valid, Y_valid, 0)
        _ = test_LR(Final_model_NT, X_test, Y_test, X_train, Y_train)
        
        #Saving the model and params
        PARAM = (Final_model_NT, X, Y, X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy(), X_valid.numpy(), Y_valid.numpy(), Lambda1, Lambda2, learning_rate)
        
        f = open(Model_NT_name, 'wb')
        pickle.dump(PARAM, f)
        f.close()
    
elif Read=='Robot':
    os.chdir('./Robot')
    
    accuracy, Lambda1, Lambda2 = 0.5, 0.001, 10**-13
    Model_NT_name = 'LR_Model_Robot_NT.pckl'
    input_dim, output_dim, epochs, batch_size = 24, 4, 500, 1000
    if not exists(Model_NT_name):
        X, Y = Read_data('sensor_readings_24_data.txt')
        """Normalization"""
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        """Train/test/validation set"""
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)
        X_test, X_valid, Y_test, Y_valid = train_test_split(X_test,Y_test, test_size = 0.5, random_state=0)
        
        X_train, X_test, X_valid = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(X_valid)
        Y_train, Y_test, Y_valid = torch.LongTensor(Y_train), torch.LongTensor(Y_test), torch.LongTensor(Y_valid)
                
        #Training the data
        model_NT = Log_Reg(input_dim, output_dim) # Model with No Transformation
        CEF_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_NT.parameters(), lr=learning_rate)
        Final_model_NT = train_LR_PPS(model_NT, epochs, optimizer, CEF_loss, Lambda1, Lambda2, batch_size, 
                                  X_train, Y_train, X_valid, Y_valid, 0)
        _ = test_LR(Final_model_NT, X_test, Y_test, X_train, Y_train)
        
        #Saving the model and params
        PARAM = (Final_model_NT, X, Y, X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy(), X_valid.numpy(), Y_valid.numpy(), Lambda1, Lambda2, learning_rate)
        
        f = open(Model_NT_name, 'wb')
        pickle.dump(PARAM, f)
        f.close()
    

elif Read=='Synthetic':
    os.chdir('./Synthetic')
    
    accuracy, Lambda1, Lambda2 = 0.5, 0.0001, 10**-13
    """First we train the data without any transformation"""
    Model_NT_name = 'LR_Model_Synthetic_NT.pckl'    
    input_dim, output_dim, epochs, batch_size = 5, 2, 500, 1000
    if not exists(Model_NT_name):
        X,Y = make_classification(n_samples=50000, n_features=input_dim, n_informative=1, n_redundant=0, n_repeated=0, 
                                  n_classes=output_dim, n_clusters_per_class=1, weights=None, 
                                  flip_y=0.1, class_sep=1.0, hypercube=True, 
                                  shift=1.0, scale=1.0, shuffle=True, random_state=0)
        
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        

        #Train/test/validation set
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)
        X_test, X_valid, Y_test, Y_valid = train_test_split(X_test,Y_test, test_size = 0.5, random_state=0)
        
        X_train, X_test, X_valid = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(X_valid)
        Y_train, Y_test, Y_valid = torch.LongTensor(Y_train), torch.LongTensor(Y_test), torch.LongTensor(Y_valid)
        
        #Training the data
        model_NT = Log_Reg(input_dim, output_dim) # Model with No Transformation
        CEF_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_NT.parameters(), lr=learning_rate)
        Final_model_NT = train_LR_PPS(model_NT, epochs, optimizer, CEF_loss, Lambda1, Lambda2, batch_size, 
                                  X_train, Y_train, X_valid, Y_valid, 0)
        _ = test_LR(Final_model_NT, X_test, Y_test, X_train, Y_train)
        
        #Saving the model and params
        PARAM = (Final_model_NT, X, Y, X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy(), X_valid.numpy(), Y_valid.numpy(), Lambda1, Lambda2, learning_rate)
                
        f = open(Model_NT_name, 'wb')
        pickle.dump(PARAM, f)
        f.close()


LOSS = 'ESA'

"""Results for the model without Linear transformation"""
f = open(Model_NT_name, 'rb')
model_data = pickle.load(f)
f.close()

Weights = np.array(list(model_data[0].named_parameters())[0][1].detach().cpu(), dtype = np.float64)
Biases = np.array(list(model_data[0].named_parameters())[1][1].detach().cpu(), dtype = np.float64)
X, Y = model_data[1], model_data[2]



#STR = 'Bef_Trans_'
#MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_NoClamp.pckl', STR, 'no', 'Non_extension')
#MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp_Extended.pckl', STR, 'yes', 'Extended')
#STR = 'PPS3_Trans_'
#MSE = PPS3(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_NoClamp.pckl', STR, 'no', 'Non_extension', epochs, batch_size, accuracy, Model_NT_name, 0)
#MSE = PPS3(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp_Extended.pckl', STR, 'yes', 'Extended', epochs, batch_size, accuracy, Model_NT_name, 0)
#
#
#STR = 'Bef_Trans_'
#MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp.pckl', STR, 'yes', 'Non_extension')
#MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_0centre.pckl', STR, 'no', '0centre')
#MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_RCC.pckl', STR, 'no', 'RCC')
#MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_CLS.pckl', STR, 'no', 'CLS')
#
STR = 'PPS3_Trans_'
MSE = PPS3(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp.pckl', STR, 'yes', 'Non_extension', epochs, batch_size, accuracy, Model_NT_name, 0)
#MSE = PPS3(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_0centre.pckl', STR, 'no', '0centre', epochs, batch_size, accuracy, Model_NT_name, 0)
#MSE = PPS3(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_RCC.pckl', STR, 'no', 'RCC', epochs, batch_size, accuracy, Model_NT_name, 0)
#MSE = PPS3(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_CLS.pckl', STR, 'no', 'CLS', epochs, batch_size, accuracy, Model_NT_name, 0)
