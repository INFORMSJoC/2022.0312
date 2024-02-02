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
# import sklearn

import pickle
import time
import random

from pyDOE import *



import numpy as np
from gurobipy import GRB

import itertools

from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from skopt.space import Space
from skopt.sampler import Lhs

from global_benchmark_functions import *
from synthetic_data_generation import *
from mahalanobis_distance_calculator import MDistance_Squared_SinglePoint
from pca_distance_calculator import *
from predictive_modeling import *
from isolation_forest import *
from volume_constraint_learning import *
from optimization_models import *
from optimization_benchmark_models import *
from optimization_svm_model_baron import *

#%%
'''
Setting of Paremeters in Experiments
'''
time_limit = 900

# Benchmark Functions (Section 4.1)
list_true_function = [beale, peaks, griewank, powell,quintic, qing, rastrigin] 

# Data Generation (Section 4.2)
global_minimum_point_dict = {'ackley': [0, 0],
                             'beale': [3, 0.5],
                             'chung': [0]*6,
                             'eggcrate': [0, 0],
                             'exponential': [0, 0],
                             'griewank': [0] * 4,
                             'mishra_11': [0] * 8,
                             'paviani': [9.35] * 10,
                             'peaks': [0.228, -1.626],
                             'peaks_adj': [0.228, -1.626],
                             'powell': [0] * 4,
                             'qing': [1, np.sqrt(2), np.sqrt(3), np.sqrt(4), np.sqrt(5), np.sqrt(6), np.sqrt(7), np.sqrt(8)],
                             'quintic': [2] * 5,
                             'rastrigin': [0] * 10,
                             'salomon': [0, 0],
                             'salomon_4': [0] * 4,
                             'wayburn_seader_1': [1, 2]
                             }
random_states = list(range(1,11,1)) # 10 different datasets, each generated with a different randomly drawn convariance matrix
n_samples = 1000
lambdas_std_noise =  [1] # related to the noise included in the outcome of each sampled point 
    
# Predictive Model Procedures (Section 4.3)
list_predictive_model = ['LR', 'NN', 'RF'] 

# Optimization Methods and Experimental Parameters (Section 4.4)
## KNN
quantil_y = 0.9
pct_good_samples = 0.1
k_neighbors = 1
## IF
number_of_isolation_trees = 100 # isolation forest
## SVM-BC
x_segment_count = 30 # m (number of pieces) in svm gurobi-callback

# Tightness Level (Section 4.5)
alph_pca_lb = 0.00001 # loosest
alph_pca_ub = 0.49999 # tightest

alph_md_lb = 0.00001 # loosest
alph_md_ub = 0.99999 # tightest

rho_knn_ub = 0.1 # loosest
rho_knn_lb = 0.0001 # tightest

Ls_if = [0,1,2,3,4,5,6] # break points of depth in IF 


#%%

start_time = datetime.datetime.now()
print('Start Time', start_time)
ExpResult = [] 
for true_function in list_true_function:     
    for a in range(len(lambdas_std_noise)):   
        AvgCorrExpectation_BestSample = 0
        sd_noise = lambdas_std_noise[a]
        for b in range(len(random_states)):      
            # Section 4.2            
            # generate a multivariate normal distribution 
            random_state = random_states[b] 
            mu = global_minimum_point_dict[true_function.__name__]
            n_dimensions = len(mu)
            Cov = sklearn.datasets.make_spd_matrix(n_dimensions, random_state = random_state) # 10 different randomly drawn covariance matrix
            np.random.seed(1)
            X = np.random.multivariate_normal(mu, Cov, 10000)
            # randomly select a point as the centroid of the synthetic data
            random.seed(1)
            mu = random.choice(X) 
            # generate a dataset consisting of 1000 points, scale to [0, 1], and split in a training set and a test set
            x_simulated, y_simulated, y_expectation, BestSampleWithNoise, CorrExpectation_BestSample = SimulatedDataset(sd_noise, random_state, true_function, n_dimensions, n_samples, mu)
            X_scaled, y_scaled, X_train, X_test, y_train, y_test, min_max_scaler, standard_scaler_y, CorrExpectation_BestSample = PreprocessData(x_simulated, y_simulated, true_function)
            
            # Section 4.3
            # the feasible range of decision variables is [0, 1] as the independent variables are scaled to [0, 1]
            LB = np.zeros(n_dimensions) 
            UB = np.ones(n_dimensions)
            
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('+++++++++++++++++++++++++++++++++++++  NewDataset  ++++++++++++++++++++++++++++++++++++++')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('benchmark function: ' + str(true_function.__name__))
            print('random state: ' + str(random_state))
            print('CorrExpectation_BestSample in Simulated Data: ' + str(CorrExpectation_BestSample))
            print('CorrExpectation_BestSample in X_train: ' + str(CorrExpectation_BestSample))
            
            # Section 4.5
            # volume: meausre the level of tightness of the different trust-region constraints
            # randomly sampling 10,000 points from a unit hypercube and calculating volume 
            space = Space([(0.0, 1.0)] * n_dimensions)       
            lhs = Lhs(lhs_type = "classic", criterion = None)
            np.random.seed(random_state)        
            X = np.array(lhs.generate(space.dimensions, 10000))
            
            # Section 4.4
            # specify trust-region constraints (IF & SVM / SVM-BC only, the remaining types of constraints are embeded into an OPPM model directly)
            ## IF
            isolation_forest_info = isolation_forest_info_reader(X_train, number_of_isolation_trees)
            ## SVM / SVM-BC
            clf = isolation_forest_info['model']
            svm = OneClassSVM(gamma='auto').fit(X_train) 
            sv = svm.support_vectors_.tolist()
            alpha = svm.dual_coef_.ravel().tolist()
            gamma = svm._gamma
            rho = -svm.intercept_.item()
            svm_score = []
            for p in range(len(X_train)):                 
                svm_score.append(svm.score_samples(X_train[p].reshape(1, -1)))
            ### Tightness Level for SVM / SVM-BC (Section 4.5)
            rho_lb = np.min(svm_score)
            rho_ub = np.max(svm_score)
            
            
            for predictive_model in list_predictive_model:  
## LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR ##
## LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR #### LR ##
                if predictive_model == 'LR':
                    lr_info = lr_info_reader(X_train, y_train, X_test, y_test)
                    TrainRsquare = lr_info['train_r_sq'] 
                    TestRsquare = lr_info['test_r_sq']
                    print('================  unrestricted result  ================')
                    OptimalObjective_unrestricted, CorrExpectation_unrestricted, Runtime_unrestricted = lr_unrestricted(lr_info, true_function, X_train, min_max_scaler, GRB.MINIMIZE, OutputFlag = 0, timelimit = time_limit)
                    OptimalObjective_unrestricted = standard_scaler_y.inverse_transform([[OptimalObjective_unrestricted]])[0][0]
                    MIPGap_unrestricted = 0
                    print("OptimalObjective_unrestricted: " + str(OptimalObjective_unrestricted))
                    print("CorrExpectation_unrestricted: " + str(CorrExpectation_unrestricted))
                    list_info_save = [true_function.__name__, random_state,'LR', 'Unrestricted', 1, 'Unrestricted', 'Unrestricted', TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_unrestricted, CorrExpectation_unrestricted, Runtime_unrestricted, 0]
                    ExpResult.append(list_info_save) 
                
                    print('================  result with isolation forest -- different depth  ================')
                    for i in range(len(Ls_if)):
                        if i == 0:
                            list_info_save = [true_function.__name__, random_state,'LR', 'IF', i+1, Ls_if[i], 1, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_unrestricted, CorrExpectation_unrestricted, Runtime_unrestricted, 0]
                            ExpResult.append(list_info_save)                            
                        else:
                            print("rho_new: " + str(Ls_if[i]))
                            volume_of_IF = volume_IF(X, clf, Ls_if[i])
                            print("volume_IF: " + str(volume_of_IF))
                            OptimalObjective_if, CorrExpectation_if, Runtime_if, MIPGap_if = lr_isolation_forest(lr_info, isolation_forest_info, true_function, X_train, min_max_scaler, Ls_if[i], objective = GRB.MINIMIZE, bigM_if=10000.0, OutputFlag = 0, timelimit = time_limit)
                            if OptimalObjective_if != 'null':
                                OptimalObjective_if = standard_scaler_y.inverse_transform([[OptimalObjective_if]])[0][0]
                            print("OptimalObjective_if: " + str(OptimalObjective_if))
                            print("CorrExpectation_if: " + str(CorrExpectation_if))
                            print("Runtime_if: " + str(Runtime_if))
                            list_info_save = [true_function.__name__, random_state,'LR', 'IF', i+1, Ls_if[i], volume_of_IF, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted,OptimalObjective_if, CorrExpectation_if, Runtime_if, MIPGap_if]
                            ExpResult.append(list_info_save) 
                            
 
                        
                    print('================  result with svm -- Baron -- different rho ================')                    
                    rho_segment = (rho_ub - rho_lb) / 6
                    rho_breakpoints = [rho_lb + rho_segment * i for i in range(7)]
                    rho_breakpoints = [trunc(number, 4) for number in rho_breakpoints]
                    for i in range(len(rho_breakpoints)):
                        rho_new = rho_breakpoints[i]
                        print("rho_new: " + str(rho_new))
                        volume_of_SVM = volume_SVM(X, svm, rho_new)
                        print("volume_SVM: " + str(volume_of_SVM))
                        OptimalObjective_svm, CorrExpectation_svm, Runtime_svm = lr_svm_baron_rho_new(svm, lr_info, rho_new, true_function, X_train, min_max_scaler, timelimit = time_limit, objective = minimize)
                        if OptimalObjective_svm != 'null':
                            OptimalObjective_svm = standard_scaler_y.inverse_transform([[OptimalObjective_svm]])[0][0]
                            MIPGap_svm = Baron_MIPGap()
                        else:
                            MIPGap_svm = 'null'
                        print('================  result with svm -- Baron  ================')
                        print("OptimalObjective_svm_baron: " + str(OptimalObjective_svm))
                        print("CorrExpectation_svm_baron: " + str(CorrExpectation_svm))
                        print("Runtime_svm_baron: " + str(Runtime_svm))
                        print("MIPGap_svm_baron: " + str(MIPGap_svm))
                        list_info_save = [true_function.__name__, random_state,'LR', 'SVM', i+1, rho_new, volume_of_SVM, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted,OptimalObjective_svm, CorrExpectation_svm, Runtime_svm, MIPGap_svm]
                        ExpResult.append(list_info_save)

                        print('================  result with svm -- Gurobi  ================')
                        OptimalObjective_svm_cb, CorrExpectation_svm_cb, Runtime_svm_cb, MIPGap_svm_cb = lr_svm_w_callback(svm, rho_new, lr_info, true_function, X_train, min_max_scaler, x_segment_count, objective = GRB.MINIMIZE, OutputFlag = 0, timelimit = time_limit)
                        if OptimalObjective_svm_cb != 'null':
                            OptimalObjective_svm_cb = standard_scaler_y.inverse_transform([[OptimalObjective_svm_cb]])[0][0]
                        # MIPGap_svm_cb = 0
                        print("OptimalObjective_svm_cb: " + str(OptimalObjective_svm_cb))
                        print("CorrExpectation_svm_cb: " + str(CorrExpectation_svm_cb))
                        print("Runtime_svm_cb: " + str(Runtime_svm_cb)) 
                        print("MIPGap_svm_cb: " + str(MIPGap_svm_cb)) 
                        list_info_save = [true_function.__name__, random_state,'LR', 'SVM-BC', i+1, rho_new, volume_of_SVM, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_svm_cb, CorrExpectation_svm_cb, Runtime_svm_cb, MIPGap_svm_cb]
                        ExpResult.append(list_info_save)
                        
                    
                    print('================  result with convex hull  ================')
                    start_time_tmp = datetime.datetime.now()
                    print('Start Time: ', start_time_tmp) 
                    volume_of_CH = volume_CH(X, X_train)
                    print("volume: " + str(volume_of_CH))
                    OptimalObjective_ch, CorrExpectation_ch, Runtime_ch = lr_convex_hull(lr_info, true_function, X_train, min_max_scaler, objective = GRB.MINIMIZE, OutputFlag = 0, timelimit = time_limit)
                    OptimalObjective_ch = standard_scaler_y.inverse_transform([[OptimalObjective_ch]])[0][0]
                    MIPGap_ch = 0
                    print("OptimalObjective_ch: " + str(OptimalObjective_ch))
                    print("CorrExpectation_ch: " + str(CorrExpectation_ch))
                    print("Runtime_ch: " + str(Runtime_ch))
                    list_info_save = [true_function.__name__, random_state, 'LR', 'CH', 1, 'CH', volume_of_CH, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted,  OptimalObjective_ch, CorrExpectation_ch, Runtime_ch, MIPGap_ch]
                    ExpResult.append(list_info_save)  
                    
                    print('================  result with pca -constraint  ================')                      
                    alpha_segment = (alph_pca_ub - alph_pca_lb) / 6
                    alpha_breakpoints = [alph_pca_lb + alpha_segment * i for i in range(7)]
                    for i in range(len(alpha_breakpoints)):
                        rho_new = PCA_Orthogonal_Distance_cutoff_value(X_train, alpha_breakpoints[i])
                        rho_new = trunc(rho_new, 4)
                        print("rho_new: " + str(rho_new))
                        volume_of_PCA = volume_PCA(X, X_train, rho_new)
                        print("volume: " + str(volume_of_PCA))
                        OptimalObjective_pca, CorrExpectation_pca, Runtime_pca = lr_pca_constr(lr_info, true_function, X_train, min_max_scaler, rho_new, objective = GRB.MINIMIZE, OutputFlag = 0, timelimit = time_limit)
                        OptimalObjective_pca = standard_scaler_y.inverse_transform([[OptimalObjective_pca]])[0][0]
                        print("CorrExpectation_pca: " + str(CorrExpectation_pca))
                        list_info_save = [true_function.__name__, random_state, 'LR', 'PCA', i+1, rho_new, volume_of_PCA, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_pca, CorrExpectation_pca, Runtime_pca, 0]
                        ExpResult.append(list_info_save)
                         

                    print('================  result with m dist  ================') 
                    alpha_segment = (alph_md_ub - alph_md_lb) / 6
                    alpha_breakpoints = [alph_md_lb + alpha_segment * i for i in range(7)]
                    for i in range(len(alpha_breakpoints)):
                        rho_new = sp.stats.chi2.ppf(1-alpha_breakpoints[i], df = n_dimensions)
                        rho_new = trunc(rho_new, 4)
                        print("rho_new: " + str(rho_new))
                        volume_of_MD = volume_MD(X, X_train, rho_new)
                        print("volume: " + str(volume_of_MD))
                        OptimalObjective_mdist, CorrExpectation_mdist, Runtime_mdist = lr_mdist(lr_info, true_function, X_train, min_max_scaler, rho_new, objective = GRB.MINIMIZE, OutputFlag = 0, timelimit = time_limit)
                        OptimalObjective_mdist = standard_scaler_y.inverse_transform([[OptimalObjective_mdist]])[0][0]
                        print("CorrExpectation_mdist: " + str(CorrExpectation_mdist))
                        list_info_save = [true_function.__name__, random_state, 'LR', 'MD', i+1, rho_new, volume_of_MD, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_mdist, CorrExpectation_mdist, Runtime_mdist, 0]
                        ExpResult.append(list_info_save)


                    print('================  result with KNN  ================')
                    X_good = TrainingDataForKNN(X_train, y_train, n_samples, pct_good_samples, lr_info['model'])
                    rho_segment = (rho_knn_ub - rho_knn_lb) / 6
                    rho_breakpoints = [rho_knn_lb + rho_segment * i for i in range(7)]
                    rho_breakpoints = [trunc(number, 4) for number in rho_breakpoints]
                    for i in range(len(rho_breakpoints)):
                        rho_new = rho_breakpoints[6-i]
                        print("rho_new: " + str(rho_new))
                        volume_of_KNN = volume_KNN(X, X_train, rho_new)
                        print("volume: " + str(volume_of_KNN))
                        D = rho_new * n_dimensions
                        OptimalObjective_kNN, CorrExpectation_kNN, Runtime_kNN, MIPGap_kNN = lr_knn(lr_info, true_function, X_good, min_max_scaler, k_neighbors, D, objective = GRB.MINIMIZE, bigM_kNN=10000.0, OutputFlag = 0, timelimit = time_limit)
                        OptimalObjective_kNN = standard_scaler_y.inverse_transform([[OptimalObjective_kNN]])[0][0]
                        print("CorrExpectation_kNN: " + str(CorrExpectation_kNN))
                        list_info_save = [true_function.__name__, random_state, 'LR', 'KNN', i+1, rho_new, volume_of_KNN, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_kNN, CorrExpectation_kNN, Runtime_kNN, MIPGap_kNN]
                        ExpResult.append(list_info_save)

                              
                
## NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN ##
## NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN #### NN ##
                elif predictive_model == 'NN':
                    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&') 
                    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')                     
                    print('benchmark function: ' + str(true_function.__name__))
                    print('random state: ' + str(random_state))
                    nn, TrainRsquare, TestRsquare = NN_cross_validation(X_train, y_train, X_test, y_test)
                    nn_info = extract_nn_info(nn)
                    print('================  unrestricted result  ================')
                    OptimalObjective_unrestricted, CorrExpectation_unrestricted, Runtime_unrestricted, MIPGap_unrestricted = nn_unrestricted(true_function, min_max_scaler, X_train, nn_info, LB, UB, bigM = 10000.0, OutputFlag = 0, timelimit = time_limit)
                    if OptimalObjective_unrestricted != 'null':
                        OptimalObjective_unrestricted = standard_scaler_y.inverse_transform([[OptimalObjective_unrestricted]])[0][0]
                    print("OptimalObjective_unrestricted: " + str(OptimalObjective_unrestricted))
                    print("CorrExpectation_unrestricted: " + str(CorrExpectation_unrestricted))
                    print("Runtime_unrestricted: " + str(Runtime_unrestricted))
                    list_info_save = [true_function.__name__, random_state, 'NN', 'Unrestricted', 1, 'Unrestricted', 'Unrestricted', TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted,  OptimalObjective_unrestricted, CorrExpectation_unrestricted, Runtime_unrestricted, MIPGap_unrestricted]
                    ExpResult.append(list_info_save) 
                        
                    print('================  result with isolation forest -- different depth  ================')                  
                    for i in range(len(Ls_if)):
                        if i == 0:
                            list_info_save = [true_function.__name__, random_state, 'NN', 'IF', i+1, Ls_if[i], 1, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted,  OptimalObjective_unrestricted, CorrExpectation_unrestricted, Runtime_unrestricted, MIPGap_unrestricted]
                            ExpResult.append(list_info_save)  

                        else:
                            print("rho_new: " + str(Ls_if[i]))
                            volume_of_IF = volume_IF(X, clf, Ls_if[i])
                            print("volume_IF: " + str(volume_of_IF))
                            OptimalObjective_if, CorrExpectation_if, Runtime_if, MIPGap_if = nn_isolation_forest(nn_info, isolation_forest_info, true_function, X_train, LB, UB, min_max_scaler, Ls_if[i], objective = GRB.MINIMIZE, bigM_if=10000.0, bigM = 10000.0, OutputFlag = 0, timelimit = time_limit)      
                            if OptimalObjective_if != 'null':
                                OptimalObjective_if = standard_scaler_y.inverse_transform([[OptimalObjective_if]])[0][0]
                            print("OptimalObjective_if: " + str(OptimalObjective_if))
                            print("CorrExpectation_if: " + str(CorrExpectation_if))
                            print("Runtime_if: " + str(Runtime_if))
                            list_info_save = [true_function.__name__, random_state, 'NN', 'IF', i+1, Ls_if[i], volume_of_IF, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted,OptimalObjective_if, CorrExpectation_if, Runtime_if, MIPGap_if]
                            ExpResult.append(list_info_save) 

     
                    print('================  result with svm -- Baron -- different rho ================')
                    rho_segment = (rho_ub - rho_lb) / 6
                    rho_breakpoints = [rho_lb + rho_segment * i for i in range(7)]
                    rho_breakpoints = [trunc(number, 4) for number in rho_breakpoints]
                    for i in range(len(rho_breakpoints)):
                        rho_new = rho_breakpoints[i]
                        print("rho_new: " + str(rho_new))
                        volume_of_SVM = volume_SVM(X, svm, rho_new)
                        print("volume_SVM: " + str(volume_of_SVM))
                        OptimalObjective_svm, CorrExpectation_svm, Runtime_svm = nn_svm_baron_rho_new(svm, nn_info, rho_new, true_function, X_train, LB, UB, min_max_scaler, objective = minimize, bigM = 10000.0, timelimit = time_limit)
                        if OptimalObjective_svm != 'null':
                            OptimalObjective_svm = standard_scaler_y.inverse_transform([[OptimalObjective_svm]])[0][0]
                            MIPGap_svm = Baron_MIPGap()
                        else:
                            MIPGap_svm = 'null'
                        print('================  result with svm -- Baron ================') 
                        print("OptimalObjective_svm_baron: " + str(OptimalObjective_svm))
                        print("CorrExpectation_svm_baron: " + str(CorrExpectation_svm))
                        print("Runtime_svm_baron: " + str(Runtime_svm))
                        print("MIPGap_svm_baron: " + str(MIPGap_svm))
                        list_info_save = [true_function.__name__, random_state, 'NN', 'SVM', i+1, rho_new, volume_of_SVM, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted,OptimalObjective_svm, CorrExpectation_svm, Runtime_svm, MIPGap_svm]
                        ExpResult.append(list_info_save) 

                        print('================  result with svm -- Gurobi callback ================')    
                        OptimalObjective_svm_cb, CorrExpectation_svm_cb, Runtime_svm_cb, MIPGap_svm_cb = nn_svm_w_callback(svm, rho_new, nn_info, true_function, X_train, LB, UB, min_max_scaler, x_segment_count, objective = GRB.MINIMIZE, bigM = 10000.0, OutputFlag = 0, timelimit = time_limit)
                        if OptimalObjective_svm_cb != 'null':
                            OptimalObjective_svm_cb = standard_scaler_y.inverse_transform([[OptimalObjective_svm_cb]])[0][0]
                        print("OptimalObjective_svm_cb: " + str(OptimalObjective_svm_cb))
                        print("CorrExpectation_svm_cb: " + str(CorrExpectation_svm_cb))
                        print("Runtime_svm_cb: " + str(Runtime_svm_cb))
                        print("MIPGap_svm_cb: " + str(MIPGap_svm_cb))
                        list_info_save = [true_function.__name__, random_state, 'NN', 'SVM-BC', i+1, rho_new, volume_of_SVM, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_svm_cb, CorrExpectation_svm_cb, Runtime_svm_cb, MIPGap_svm_cb]
                        ExpResult.append(list_info_save) 

                    
                    print('================  result with convex hull  ================')
                    OptimalObjective_ch, CorrExpectation_ch, Runtime_ch, MIPGap_ch = nn_convex_hull(nn_info, true_function, X_train, LB, UB, min_max_scaler, objective = GRB.MINIMIZE, bigM = 10000.0, OutputFlag = 0, timelimit = time_limit)
                    OptimalObjective_ch = standard_scaler_y.inverse_transform([[OptimalObjective_ch]])[0][0]
                    print("CorrExpectation_ch: " + str(CorrExpectation_ch))
                    print("Runtime_ch: " + str(Runtime_ch))
                    list_info_save = [true_function.__name__, random_state, 'NN', 'CH', 1, 'CH', 'CH', TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted,  OptimalObjective_ch, CorrExpectation_ch, Runtime_ch, MIPGap_ch]
                    ExpResult.append(list_info_save) 
                   
                    print('================  result with pca -constraint  ================')  
                    alpha_segment = (alph_pca_ub - alph_pca_lb) / 6
                    alpha_breakpoints = [alph_pca_lb + alpha_segment * i for i in range(7)]
                    for i in range(len(alpha_breakpoints)):
                        rho_new = PCA_Orthogonal_Distance_cutoff_value(X_train, alpha_breakpoints[i])
                        rho_new = trunc(rho_new, 4)
                        print("rho_new: " + str(rho_new))
                        volume_of_PCA = volume_PCA(X, X_train, rho_new)
                        print("volume: " + str(volume_of_PCA))
                        OptimalObjective_pca, CorrExpectation_pca, Runtime_pca, MIPGap_pca = nn_pca_constr(nn_info, true_function, X_train, LB, UB, min_max_scaler, rho_new, objective = GRB.MINIMIZE, bigM = 10000.0, OutputFlag = 0, timelimit = time_limit)
                        OptimalObjective_pca = standard_scaler_y.inverse_transform([[OptimalObjective_pca]])[0][0]
                        print("CorrExpectation_pca: " + str(CorrExpectation_pca))
                        list_info_save = [true_function.__name__, random_state, 'NN', 'PCA', i+1, rho_new, volume_of_PCA, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_pca, CorrExpectation_pca, Runtime_pca, 0]
                        ExpResult.append(list_info_save)


                    print('================  result with m dist  ================') 
                    alpha_segment = (alph_md_ub - alph_md_lb) / 6
                    alpha_breakpoints = [alph_md_lb + alpha_segment * i for i in range(7)]
                    for i in range(len(alpha_breakpoints)):
                        rho_new = sp.stats.chi2.ppf(1-alpha_breakpoints[i], df = n_dimensions)
                        rho_new = trunc(rho_new, 4)
                        print("rho_new: " + str(rho_new))
                        volume_of_MD = volume_MD(X, X_train, rho_new)
                        print("volume: " + str(volume_of_MD))
                        OptimalObjective_mdist, CorrExpectation_mdist, Runtime_mdist, MIPGap_mdist = nn_mdist(true_function, min_max_scaler, X_train, nn_info, LB, UB, rho_new, OutputFlag = 0, bigM = 10000.0, timelimit = time_limit)
                        OptimalObjective_mdist = standard_scaler_y.inverse_transform([[OptimalObjective_mdist]])[0][0]
                        print("CorrExpectation_mdist: " + str(CorrExpectation_mdist))
                        list_info_save = [true_function.__name__, random_state, 'NN', 'MD', i+1, rho_new, volume_of_MD, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_mdist, CorrExpectation_mdist, Runtime_mdist, 0]
                        ExpResult.append(list_info_save)


                    print('================  result with KNN  ================')
                    X_good = TrainingDataForKNN(X_train, y_train, n_samples, pct_good_samples, nn)
                    rho_segment = (rho_knn_ub - rho_knn_lb) / 6
                    rho_breakpoints = [rho_knn_lb + rho_segment * i for i in range(7)]
                    rho_breakpoints = [trunc(number, 4) for number in rho_breakpoints]
                    for i in range(len(rho_breakpoints)):
                        rho_new = rho_breakpoints[6-i]
                        print("rho_new: " + str(rho_new))
                        volume_of_KNN = volume_KNN(X, X_train, rho_new)
                        print("volume: " + str(volume_of_KNN))
                        D = rho_new * n_dimensions
                        OptimalObjective_kNN, CorrExpectation_kNN, Runtime_kNN, MIPGap_kNN = nn_kNN(true_function, min_max_scaler, X_good, k_neighbors, D, nn_info, LB, UB, bigM_kNN = 10000, bigM = 10000, OutputFlag = 0, timelimit = time_limit)
                        OptimalObjective_kNN = standard_scaler_y.inverse_transform([[OptimalObjective_kNN]])[0][0]
                        print("CorrExpectation_kNN: " + str(CorrExpectation_kNN))
                        list_info_save = [true_function.__name__, random_state, 'NN', 'KNN', i+1, rho_new, volume_of_KNN, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_kNN, CorrExpectation_kNN, Runtime_kNN, MIPGap_kNN]
                        ExpResult.append(list_info_save)

    
## RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF ##
## RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF #### RF ##
                elif predictive_model == 'RF':
                    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&') 
                    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&') 
                    print('benchmark function: ' + str(true_function.__name__))
                    print('random state: ' + str(random_state))
                    rf_info = rf_info_reader_v1(X_train, y_train, X_test, y_test) # cross validation embeded
                    print("parameters in random forest: " + str(rf_info['params']))
                    TrainRsquare = rf_info['train_r_sq']  
                    TestRsquare = rf_info['test_r_sq']
                    print('================  unrestricted result  ================')       
                    OptimalObjective_unrestricted, CorrExpectation_unrestricted, Runtime_unrestricted, MIPGap_unrestricted = rf_unrestricted(rf_info, true_function, X_train, min_max_scaler, GRB.MINIMIZE, OutputFlag = 0, timelimit = time_limit)
                    OptimalObjective_unrestricted = standard_scaler_y.inverse_transform([[OptimalObjective_unrestricted]])[0][0]
                    print("CorrExpectation_unrestricted: " + str(CorrExpectation_unrestricted))
                    print("Runtime_unrestricted: " + str(Runtime_unrestricted))
                    list_info_save = [true_function.__name__, random_state, 'RF', 'Unrestricted', 1, 'Unrestricted', 'Unrestricted', TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_unrestricted, CorrExpectation_unrestricted, Runtime_unrestricted, MIPGap_unrestricted]
                    ExpResult.append(list_info_save) 

                        
                    print('================  result with isolation forest -- different depth  ================')                                    
                    for i in range(len(Ls_if)):
                        if i == 0:
                            list_info_save = [true_function.__name__, random_state, 'RF', 'IF', i+1, Ls_if[i], 1, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_unrestricted, CorrExpectation_unrestricted, Runtime_unrestricted, MIPGap_unrestricted]
                            ExpResult.append(list_info_save)                            
                        else:
                            print("rho_new: " + str(Ls_if[i]))
                            volume_of_IF = volume_IF(X, clf, Ls_if[i])
                            print("volume_IF: " + str(volume_of_IF))
                            OptimalObjective_if, CorrExpectation_if, Runtime_if, MIPGap_if = rf_isolation_forest(rf_info, isolation_forest_info, true_function, X_train, min_max_scaler, Ls_if[i], objective = GRB.MINIMIZE, bigM_if=10000.0, OutputFlag = 0, timelimit = time_limit)
                            if OptimalObjective_if != 'null':
                                OptimalObjective_if = standard_scaler_y.inverse_transform([[OptimalObjective_if]])[0][0]
                            print("OptimalObjective_if: " + str(OptimalObjective_if))
                            print("CorrExpectation_if: " + str(CorrExpectation_if))
                            print("Runtime_if: " + str(Runtime_if))
                            list_info_save = [true_function.__name__, random_state, 'RF', 'IF', i+1, Ls_if[i], volume_of_IF, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted,OptimalObjective_if, CorrExpectation_if, Runtime_if, MIPGap_if]
                            ExpResult.append(list_info_save) 

  
                    print('================  result with svm -- Baron -- different rho ================')
                    rho_segment = (rho_ub - rho_lb) / 6
                    rho_breakpoints = [rho_lb + rho_segment * i for i in range(7)]
                    rho_breakpoints = [trunc(number, 4) for number in rho_breakpoints]
                    for i in range(len(rho_breakpoints)):
                        rho_new = rho_breakpoints[i]
                        print("rho_new: " + str(rho_new))
                        volume_of_SVM = volume_SVM(X, svm, rho_new)
                        print("volume_SVM: " + str(volume_of_SVM))
                        OptimalObjective_svm, CorrExpectation_svm, Runtime_svm = rf_svm_baron_rho_new(svm, rf_info, rho_new, true_function, X_train, min_max_scaler, objective = minimize, timelimit = time_limit)
                        if OptimalObjective_svm != 'null':
                            OptimalObjective_svm = standard_scaler_y.inverse_transform([[OptimalObjective_svm]])[0][0]
                            MIPGap_svm = Baron_MIPGap()
                        else:
                            MIPGap_svm = 'null'
                        print('================  result with svm -- Baron ================')    
                        print("OptimalObjective_svm_baron: " + str(OptimalObjective_svm))
                        print("CorrExpectation_svm_baron: " + str(CorrExpectation_svm))
                        print("Runtime_svm_baron: " + str(Runtime_svm))
                        print("MIPGap_svm_baron: " + str(MIPGap_svm))
                        list_info_save = [true_function.__name__, random_state, 'RF', 'SVM', i+1, rho_new, volume_of_SVM, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted,OptimalObjective_svm, CorrExpectation_svm, Runtime_svm, MIPGap_svm]
                        ExpResult.append(list_info_save) 
                                    
                        print('================  result with svm -- Gurobi callback ================')    
                        OptimalObjective_svm_cb, CorrExpectation_svm_cb, Runtime_svm_cb, MIPGap_svm_cb = rf_svm_w_callback(svm, rho_new, rf_info, true_function, X_train, min_max_scaler, x_segment_count, objective = GRB.MINIMIZE, OutputFlag = 0, timelimit = time_limit)
                        if OptimalObjective_svm_cb != 'null':
                            OptimalObjective_svm_cb = standard_scaler_y.inverse_transform([[OptimalObjective_svm_cb]])[0][0]
                        print("OptimalObjective_svm_cb: " + str(OptimalObjective_svm_cb))
                        print("CorrExpectation_svm_cb: " + str(CorrExpectation_svm_cb))
                        print("Runtime_svm_cb: " + str(Runtime_svm_cb))
                        print("MIPGap_svm_cb: " + str(MIPGap_svm_cb))
                        list_info_save = [true_function.__name__, random_state, 'RF', 'SVM-BC', i+1, rho_new, volume_of_SVM, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_svm_cb, CorrExpectation_svm_cb, Runtime_svm_cb, MIPGap_svm_cb]
                        ExpResult.append(list_info_save) 

                        
                    print('================  result with convex hull  ================')
                    OptimalObjective_ch, CorrExpectation_ch, Runtime_ch, MIPGap_ch = rf_convex_hull(rf_info, true_function, X_train, min_max_scaler, objective = GRB.MINIMIZE, OutputFlag = 0, timelimit = time_limit)
                    OptimalObjective_ch = standard_scaler_y.inverse_transform([[OptimalObjective_ch]])[0][0]
                    print("CorrExpectation_convex_hull: " + str(CorrExpectation_ch))
                    print("Runtime_convex_hull: " + str(Runtime_ch))
                    list_info_save = [true_function.__name__, random_state, 'RF', 'CH', 1, 'CH', 'CH', TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted,  OptimalObjective_ch, CorrExpectation_ch, Runtime_ch, MIPGap_ch]
                    ExpResult.append(list_info_save) 

            
                    print('================  result with pca -constraint  ================')  
                    alpha_segment = (alph_pca_ub - alph_pca_lb) / 6
                    alpha_breakpoints = [alph_pca_lb + alpha_segment * i for i in range(7)]
                    for i in range(len(alpha_breakpoints)):
                        rho_new = PCA_Orthogonal_Distance_cutoff_value(X_train, alpha_breakpoints[i])
                        rho_new = trunc(rho_new, 4)
                        print("rho_new: " + str(rho_new))
                        volume_of_PCA = volume_PCA(X, X_train, rho_new)
                        print("volume: " + str(volume_of_PCA))
                        OptimalObjective_pca, CorrExpectation_pca, Runtime_pca, MIPGap_pca = rf_pca_constr(rf_info, true_function, X_train, min_max_scaler, rho_new, objective = GRB.MINIMIZE, OutputFlag = 0, timelimit = time_limit)
                        OptimalObjective_pca = standard_scaler_y.inverse_transform([[OptimalObjective_pca]])[0][0]
                        print("CorrExpectation_pca: " + str(CorrExpectation_pca))
                        list_info_save = [true_function.__name__, random_state, 'RF', 'PCA', i+1, rho_new, volume_of_PCA, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_pca, CorrExpectation_pca, Runtime_pca, 0]
                        ExpResult.append(list_info_save)

            
                    print('================  result with m dist  ================') 
                    alpha_segment = (alph_md_ub - alph_md_lb) / 6
                    alpha_breakpoints = [alph_md_lb + alpha_segment * i for i in range(7)]
                    for i in range(len(alpha_breakpoints)):
                        rho_new = sp.stats.chi2.ppf(1-alpha_breakpoints[i], df = n_dimensions)
                        rho_new = trunc(rho_new, 4)
                        print("rho_new: " + str(rho_new))
                        volume_of_MD = volume_MD(X, X_train, rho_new)
                        print("volume: " + str(volume_of_MD))
                        OptimalObjective_mdist, CorrExpectation_mdist, Runtime_mdist, MIPGap_mdist = rf_mdist(rf_info, true_function, X_train, min_max_scaler, rho_new, objective = GRB.MINIMIZE, OutputFlag = 0, timelimit = time_limit)
                        if OptimalObjective_mdist != 'null':
                            OptimalObjective_mdist = standard_scaler_y.inverse_transform([[OptimalObjective_mdist]])[0][0]
                        print("CorrExpectation_mdist: " + str(CorrExpectation_mdist))
                        list_info_save = [true_function.__name__, random_state, 'RF', 'MD', i+1, rho_new, volume_of_MD, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_mdist, CorrExpectation_mdist, Runtime_mdist, 0]
                        ExpResult.append(list_info_save)

                    print('================  result with KNN  ================')
                    X_good = TrainingDataForKNN(X_train, y_train, n_samples, pct_good_samples, rf_info['model'])
                    rho_segment = (rho_knn_ub - rho_knn_lb) / 6
                    rho_breakpoints = [rho_knn_lb + rho_segment * i for i in range(7)]
                    rho_breakpoints = [trunc(number, 4) for number in rho_breakpoints]
                    for i in range(len(rho_breakpoints)):
                        rho_new = rho_breakpoints[6-i]
                        print("rho_new: " + str(rho_new))
                        volume_of_KNN = volume_KNN(X, X_train, rho_new)
                        print("volume: " + str(volume_of_KNN))
                        D = rho_new * n_dimensions
                        OptimalObjective_kNN, CorrExpectation_kNN, Runtime_kNN, MIPGap_kNN = rf_kNN(true_function, min_max_scaler, X_good, k_neighbors, D, rf_info, objective = GRB.MINIMIZE, bigM_kNN=10000.0, OutputFlag = 0, timelimit = time_limit)
                        OptimalObjective_kNN = standard_scaler_y.inverse_transform([[OptimalObjective_kNN]])[0][0]
                        print("CorrExpectation_kNN: " + str(CorrExpectation_kNN))
                        list_info_save = [true_function.__name__, random_state, 'RF', 'KNN', i+1, rho_new, volume_of_KNN, TrainRsquare, TestRsquare, BestSampleWithNoise, CorrExpectation_BestSample, OptimalObjective_unrestricted, CorrExpectation_unrestricted, OptimalObjective_kNN, CorrExpectation_kNN, Runtime_kNN, MIPGap_kNN]
                        ExpResult.append(list_info_save)
    
                else:
                    print('Not a predictive model supported')
                    
column_names = ["function", "random state", "predictive model", "method", "tightness level", "tightness value", "volume", "TrainRsquare", "TestRsquare", "BestSampleWithNoise", "CorrExpectation_BestSample", "OptimalObjective_unrestricted", "CorrExpectation_unrestricted", "OptimalObjective", "CorrExpectation", "Runtime", "MIPGap"] 
df = pd.DataFrame(ExpResult, columns = column_names)  
df.to_excel('exp_result_synthetic_data.xlsx', index = False) 

     
print('========== Finish ===========')
stop_time = datetime.datetime.now()
print ("Completed time : ", stop_time)
elapsed_time = stop_time - start_time
print('Elapsed Time: ', elapsed_time) 



     
print('========== Finish ===========')
stop_time = datetime.datetime.now()
print ("Completed time : ", stop_time)
elapsed_time = stop_time - start_time
print('Elapsed Time: ', elapsed_time) 