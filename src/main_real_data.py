#!/usr/bin/env python
# coding: utf-8
#%%
import sys
print(sys.path)
sys.path.insert(0, 'C:/Users/cshi/OneDrive - University of Connecticut/2019-20_Fall/First_Year_Project/OptimizationBasedOn_LR_RF_NN')

# In[17]:
import datetime
import os
import keyboard

import math
import scipy as sp
import pandas as pd 
import pickle
import time
import random
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro

from pyDOE import *

import numpy as np
from gurobipy import GRB
from mahalanobis_distance_calculator import MDistance_Squared_SinglePoint

from global_benchmark_functions import *
from predictive_modeling import *
from optimization_models import *
from isolation_forest import *
from optimization_benchmark_models import *
from optimization_svm_model_baron import *
from volume_constraint_learning import *

import itertools

from pca_distance_calculator import *
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from skopt.space import Space
from skopt.sampler import Lhs

#%%
'''
Load wine data and preprocess
'''

dataset_name = 'winequality-red.csv'
Data = pd.read_csv("./{}".format(dataset_name))
print(Data)

X = Data.iloc[:,:-1]
y = Data[Data.columns[-1:]]
y = y.values.ravel()
X_scaled = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 1)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

#%%
'''
Setting of Paremeters in Experiments (Section 4.7)
'''
# Predictive Model Procedures (Section 4.3)
list_predictive_model = [ 'LR', 'NN', 'RF']

# Optimization Methods and Experimental Parameters (Section 4.4)
# IF
number_of_isolation_trees = 100
isolation_forest_info = isolation_forest_info_reader(X_train, number_of_isolation_trees) 
clf = isolation_forest_info['model']
Ls_if = [0,1,2,3,4,5,6] # break points of depth in IF, we display results at L = 5 in the paper

# MD
alph_md_lb = 0.00001 # loosest
alph_md_ub = 0.99999 # tightest

# the feasible range of decision variables is [0, 1] as the independent variables are scaled to [0, 1]
n_dimensions = np.shape(X_train)[1]
LB = np.zeros(n_dimensions) 
UB = np.ones(n_dimensions)  

time_limit = 1800


#%%
SOL = []
start_time = datetime.datetime.now()
print('Start Time', start_time)


for predictive_model in list_predictive_model:  
## LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR ##
## LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR ##
    if predictive_model == 'LR':
        lr_info = lr_info_reader(X_train, y_train, X_test, y_test)
        TrainRsquare = lr_info['train_r_sq'] 
        TestRsquare = lr_info['test_r_sq']
        print('LR-train fit')
        print('Training R square: ', TrainRsquare)
        print('Testing R square: ', TestRsquare)
        
        print('================  unrestricted result  ================')
        OptimalObjective_unrestricted, Sol_unrestricted, Runtime_unrestricted = lr_unrestricted_real_data(lr_info, objective = GRB.MAXIMIZE, OutputFlag = 0, timelimit = time_limit)
        SOL.append(['LR'] + [TrainRsquare] + [TestRsquare] + ['Unrestricted'] + ['Unrestricted'] + ['Unrestricted'] + Sol_unrestricted)

        print('================  result with m dist  ================') 
        alpha_segment = (alph_md_ub - alph_md_lb) / 6
        alpha_breakpoints = [alph_md_lb + alpha_segment * i for i in range(7)]
        for i in range(len(alpha_breakpoints)): # we display results at i = 5 in the paper
            rho_new = sp.stats.chi2.ppf(1-alpha_breakpoints[i], df = n_dimensions)
            rho_new = trunc(rho_new, 4)
            print("rho_new: " + str(rho_new))       
            OptimalObjective_md, Sol_md, Runtime_md = lr_mdist_real_data(lr_info, X_train, min_max_scaler, rho_new, objective = GRB.MAXIMIZE, OutputFlag = 0, timelimit = time_limit)
            if type(Sol_md) != str:
                SOL.append(['LR'] + [TrainRsquare] + [TestRsquare] + ['MD'] + [i+1] + [rho_new] + Sol_md)
            else:
                SOL.append(['LR'] + [TrainRsquare] + [TestRsquare] + ['MD'] + [i+1] + [rho_new] + [Sol_md]*11)              
            
        print('================  result with isolation forest -- different depth  ================')                     
        for i in range(len(Ls_if)):
            print("rho_new: " + str(Ls_if[i]))
            L = Ls_if[i]
            OptimalObjective_if, Sol_if, Runtime_if, MIPGap_if = lr_isolation_forest_real_data(lr_info, isolation_forest_info, X_train, min_max_scaler, L, objective = GRB.MAXIMIZE, bigM_if=10000.0, OutputFlag = 0, timelimit = time_limit)
            SOL.append(['LR'] + [TrainRsquare] + [TestRsquare] + ['IF'] + [i+1] + [L] + Sol_if)

            

        
## NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN ##
## NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN ##
    elif predictive_model == 'NN':
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&') 
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')      
        nn_model_from_train_set, TrainRsquare, TestRsquare = NN_cross_validation_real_data(X_train_ = X_train, y_train_ = y_train, X_test_ = X_test, y_test_ = y_test)
        nn_info = extract_nn_info(nn_model_from_train_set)
        print('NN-train fit')
        print(nn_model_from_train_set.hidden_layer_sizes)
        print('Training R square: ', TrainRsquare)
        print('Testing R square: ', TestRsquare)
    
        print('================  unrestricted result  ================')
        OptimalObjective_unrestricted, Sol_unrestricted, Runtime_unrestricted, MIPGap_unrestricted = nn_unrestricted_real_data(nn_info, LB, UB, obj = GRB.MAXIMIZE, bigM = 100000.0, OutputFlag = 0, timelimit = time_limit)
        SOL.append(['NN'] + [TrainRsquare] + [TestRsquare] + ['Unrestricted'] + ['Unrestricted'] + ['Unrestricted'] + Sol_unrestricted.tolist())

        print('================  result with m dist  ================') 
        alpha_segment = (alph_md_ub - alph_md_lb) / 6
        alpha_breakpoints = [alph_md_lb + alpha_segment * i for i in range(7)]
        for i in range(len(alpha_breakpoints)):
            rho_new = sp.stats.chi2.ppf(1-alpha_breakpoints[i], df = n_dimensions)
            rho_new = trunc(rho_new, 4)
            print("rho_new: " + str(rho_new))       
            OptimalObjective_md, Sol_md, Runtime_md, MIPGap_md = nn_mdist_real_data(min_max_scaler, X_train, nn_info, LB, UB, rho_new, obj = GRB.MAXIMIZE, bigM = 10000.0, OutputFlag = 0, timelimit = time_limit)
            if type(Sol_md) != str:
                SOL.append(['NN'] + [TrainRsquare] + [TestRsquare] + ['MD'] + [i+1] + [rho_new] + Sol_md.tolist())
            else:
                SOL.append(['NN'] + [TrainRsquare] + [TestRsquare] + ['MD'] + [i+1] + [rho_new] + [Sol_md]*11)    


        print('================  result with isolation forest -- different depth  ================')              
        for i in range(len(Ls_if)):
            print("rho_new: " + str(Ls_if[i]))
            L = Ls_if[i]        
            OptimalObjective_if, Sol_if, Runtime_if, MIPGap_if = nn_isolation_forest_real_data(nn_info, isolation_forest_info, X_train, LB, UB, min_max_scaler, L, objective = GRB.MAXIMIZE, bigM_if=10000.0, bigM = 10000.0, OutputFlag = 0, timelimit = time_limit)
            SOL.append(['NN'] + [TrainRsquare] + [TestRsquare] + ['IF'] + [i+1] + [L] + Sol_if.tolist())



## RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF ##
## RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF ##
    elif predictive_model == 'RF':
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&') 
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&') 
        rf_info = rf_info_reader_v1(X_train_ = X_train, y_train_ = y_train, X_test_ = X_test, y_test_ = y_test)
        print('RF-train fit')
        TrainRsquare = rf_info['train_r_sq']  
        TestRsquare = rf_info['test_r_sq']
        print('Training R square: ', TrainRsquare)
        print('Testing R square: ', TestRsquare)
 
        print('================  unrestricted result  ================')
        OptimalObjective_unrestricted, Sol_unrestricted, Runtime_unrestricted, MIPGap_unrestricted = rf_unrestricted_real_data(rf_info, objective = GRB.MAXIMIZE, OutputFlag = 0, timelimit = time_limit)
        SOL.append(['RF'] + [TrainRsquare] + [TestRsquare] + ['Unrestricted'] + ['Unrestricted'] + ['Unrestricted'] + Sol_unrestricted)


        print('================  result with m dist  ================') 
        alpha_segment = (alph_md_ub - alph_md_lb) / 6
        alpha_breakpoints = [alph_md_lb + alpha_segment * i for i in range(7)]
        for i in range(len(alpha_breakpoints)):
            rho_new = sp.stats.chi2.ppf(1-alpha_breakpoints[i], df = n_dimensions)
            rho_new = trunc(rho_new, 4)
            print("rho_new: " + str(rho_new))       
            OptimalObjective_md, Sol_md, Runtime_md, MIPGap_md = rf_mdist_real_data(rf_info, X_train, min_max_scaler, rho_new, objective = GRB.MAXIMIZE, OutputFlag = 0, timelimit = time_limit)
            if type(Sol_md) != str:
                SOL.append(['RF'] + [TrainRsquare] + [TestRsquare] + ['MD'] + [i+1] + [rho_new] + Sol_md)
            else:
                SOL.append(['RF'] + [TrainRsquare] + [TestRsquare] + ['MD'] + [i+1] + [rho_new] + [Sol_md]*11)


        print('================  result with isolation forest -- different depth  ================')                     
        for i in range(len(Ls_if)):
            print("rho_new: " + str(Ls_if[i]))
            L = Ls_if[i] 
            OptimalObjective_if, Sol_if, Runtime_if, MIPGap_if = rf_isolation_forest_real_data(rf_info, isolation_forest_info, X_train, min_max_scaler, L, objective = GRB.MAXIMIZE, bigM_if=10000.0, OutputFlag = 0, timelimit = time_limit)
            SOL.append(['RF'] + [TrainRsquare] + [TestRsquare] + ['IF'] + [i+1] + [L] + Sol_if)
 
        
column_names = ["predictive model", "TrainRsquare", "TestRsquare", "Method", "Tight_Level", "Tight_Value"] + X_scaled.columns.tolist()
df = pd.DataFrame(SOL, columns = column_names)  
df.to_excel('exp_result_wine_data.xlsx', index = False) 


print('========== Finish ===========')
stop_time = datetime.datetime.now()
print ("Completed time : ", stop_time)
elapsed_time = stop_time - start_time
print('Elapsed Time: ', elapsed_time)




