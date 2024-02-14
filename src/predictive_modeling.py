#!/usr/bin/env python
# coding: utf-8


#%%
import datetime
import os
import keyboard

import math
import numpy as np
import scipy as sp
import pandas as pd 
import pickle
import time
import random
import itertools

from pyDOE import *

import sklearn.datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
min_max_scaler = MinMaxScaler()

from RF_Info_Class import Tree, LeafNode, SplitNode


#%%

def NN_cross_validation(X_train_, y_train_, X_test_, y_test_):
    """
    Train a neural network model
    """
    params = {'hidden_layer_sizes': list(itertools.product(list(range(1,11,1)), list(range(1,11,1)))),
          'solver': ['lbfgs'],
          'max_iter': [100000]}
    mlp_regressor_grid = GridSearchCV(MLPRegressor(random_state = 1), param_grid = params, n_jobs = -1, cv = 5, verbose = 5)
    mlp_regressor_grid.fit(X_train_, y_train_)
    
    nn = mlp_regressor_grid.best_estimator_
       
    TrainMSE = mean_squared_error(y_train_, nn.predict(X_train_))
    TestMSE = mean_squared_error(y_test_, nn.predict(X_test_))
    TrainRsquare = nn.score(X_train_, y_train_)
    TestRsquare = nn.score(X_test_, y_test_)
    print('NN structure: {}'.format(nn.hidden_layer_sizes))
    print ('NN_r-squared_training: {:.2f}, NN_r-squared_testing: {:.2f}' .format(TrainRsquare, TestRsquare))
    return nn, TrainRsquare, TestRsquare

#%%

def NN_cross_validation_real_data(X_train_, y_train_, X_test_, y_test_):
    """
    Train a neural network model
    """
    params = {'hidden_layer_sizes': list(itertools.product(list(range(1,11,1)), list(range(1,11,1)))),
          'solver': ['adam'],
          'max_iter': [100000]}
    mlp_regressor_grid = GridSearchCV(MLPRegressor(random_state = 5), param_grid = params, n_jobs = -1, cv = 5, verbose = 5)
    mlp_regressor_grid.fit(X_train_, y_train_)
    
    nn = mlp_regressor_grid.best_estimator_
       
    TrainMSE = mean_squared_error(y_train_, nn.predict(X_train_))
    TestMSE = mean_squared_error(y_test_, nn.predict(X_test_))
    TrainRsquare = nn.score(X_train_, y_train_)
    TestRsquare = nn.score(X_test_, y_test_)
    print ('NN_r-squared_training: {:.2f}, NN_r-squared_testing: {:.2f}' .format(TrainRsquare, TestRsquare))
    return nn, TrainRsquare, TestRsquare


#%%

def extract_nn_info(nn):
    """
    retreive information from a pre-trained neural network model
    """
    #get NN weights and bias
    NN_coefs = nn.coefs_  #list of ndarrays
    NN_intercepts = nn.intercepts_ # list of ndarrays

    n_layers = len(NN_coefs)+1 # add input layer
    layers_size = []
    for i in range(n_layers-1):
        layers_size.append(len(NN_coefs[i]))
    layers_size.append(1) #add output layer size

    offset = [] #store index of the initial and end node in each layer
    offset.append(0)
    for i in range(n_layers):
        offset.append(offset[i] + layers_size[i])
    
    n_nodes = sum(layers_size)
    
    #a sparse matrix to store weights (i,j) -- weights between node i and node j
    Ws = np.zeros((n_nodes, n_nodes), dtype = float)
    A = np.zeros((n_nodes, n_nodes), dtype = int)
    
    for layer in range(n_layers - 1):
        for i in range(layers_size[layer]):
            for j in range(layers_size[layer+1]):
                A[i + offset[layer], j + offset[layer + 1]] = 1
                Ws[i + offset[layer], j + offset[layer + 1]] = np.round(NN_coefs[layer][i][j], 4)
    N_INPUTS = layers_size[0]
    
    #store bias into 1 dimension array
    bs = list(np.zeros(N_INPUTS))
    for i in range(n_layers-1):
        bs.extend(list(NN_intercepts[i]))
    bs = np.round(np.array(bs), 4)
    
    node_layer_mapping = {}
    for i in range(n_nodes):
        layer_index_for_node = 0
        for layer_iter in range(n_layers):
            if(i >= offset[layer_iter]):
                layer_index_for_node = layer_iter
        node_layer_mapping[i] = layer_index_for_node
        
    nn_info = {'layers_size': layers_size, 'offset': offset, 'n_nodes': n_nodes, 
               'node_layer_mapping': node_layer_mapping, 'Ws': Ws, 'bs': bs}
    
    return nn_info

#%%
import math
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


#%%
    
def rf_info_reader_v1(X_train_, y_train_, X_test_, y_test_):
    """
    fit a random forest model using cross validation
    retreive information from a pre-trained RF model, mainly the leafnodes and splitnodes
    """

    # fit a random forest model
    params = {'max_depth': list(range(1,11,1)), 'n_estimators': list(range(10,110,10))}
    rf_grid = GridSearchCV(RandomForestRegressor(random_state = 2), param_grid = params, n_jobs = -1, cv = 5, verbose = 5)
    rf_grid.fit(X_train_, y_train_)
    
    rf = rf_grid.best_estimator_
    r_squared_train = round(rf.score(X_train_, y_train_), 4)
    r_squared_test = round(rf.score(X_test_, y_test_), 4)   
    print ('RF_r-squared_training: {:.2f}, RF_r-squared_testing: {:.2f}' .format(r_squared_train, r_squared_test))
 
    print('Train R^2 Score : %.3f'%rf_grid.best_estimator_.score(X_train_, y_train_))
    print('Test R^2 Score : %.3f'%rf_grid.best_estimator_.score(X_test_, y_test_))
    print('Best R^2 Score Through Grid Search : %.3f'%rf_grid.best_score_)
    print('Best Parameters : ',rf_grid.best_params_)
    
   
    # extract random forest information
    # list of LB & UB for features, 1-D list
    n_dimensions = np.shape(X_train_)[1]
    LB = np.zeros(n_dimensions) 
    UB = np.ones(n_dimensions)         
    X_train_ = pd.DataFrame(X_train_)
    # trees: leafnode_count & splitnode_count for each tree, 1-D list
    trees = {}
    # splitnodes, leafnodes: 2-D list
    splitnodes = {}
    leafnodes = {}
    
    for k in range(len(rf)):
        
        # splitnodes, leafnodes: 2-D list    
        leafnodes[k] = {}
        splitnodes[k] = {}
        
        estimator = rf.estimators_[k]
        parents = np.zeros((len(estimator.tree_.feature),2))
        parents[0] = [-1, -1] # root node parent is set to -1
        parents[0] = [-1, -1]  # root node parent is set to -1
        i = 0
        while i < len(estimator.tree_.feature):
            if estimator.tree_.children_left[i] != -1:
                parents[estimator.tree_.children_left[i]] = [i , False]
            if estimator.tree_.children_right[i] != -1:
                parents[estimator.tree_.children_right[i]] = [i , True]
            i = i + 1
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
    
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
    
            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        parents = np.zeros((len(estimator.tree_.feature),2))
        parents[0] = [-1, -1] # root node parent is set to -1
        parents[0] = [-1, -1]  # root node parent is set to -1
        i = 0
        while i < len(estimator.tree_.feature):
            if estimator.tree_.children_left[i] != -1:
                parents[estimator.tree_.children_left[i]] = [i , False]
            if estimator.tree_.children_right[i] != -1:
                parents[estimator.tree_.children_right[i]] = [i , True]
            i = i + 1
        leaf_nodes =[]
        for i in range(len(estimator.tree_.impurity)):
            if is_leaves[i] == True:
                leaf_nodes.append(i)
        
        trees[k] = Tree(len(leaf_nodes), len(leaf_nodes) - 1)
    
    
        def get_path (parents, i, path):
            j = i
            while parents[j][0] != -1:
                j = int(parents[j][0])
                path.append(j)
            return path
        features  = [X_train_.columns[i] for i in estimator.tree_.feature]
        value = estimator.tree_.value
    
    
        split = []
        for i in range(len(children_left)):
            if children_left[i] != -1:
                split.append(i)
        
        split_node_count = 0
        leaf_node_count = 0
        IndexDict = dict()
        for i in range(n_nodes):
            if  feature[i] != -2:
                IndexDict[i] = [split_node_count, 0] # 0 denote this is not a leaf node
                split_node_count = split_node_count + 1
            else:
                IndexDict[i] = leaf_node_count
                IndexDict[i] = [leaf_node_count, 1] # 1 denote this is a leaf node
                leaf_node_count = leaf_node_count + 1
            
        for i in IndexDict:  
            if  IndexDict[i][1] == 0: ## select split nodes from all the nodes
                # IndexDict[i][0]: split node index, start from 0
                # write variable, criterion, left child id, if left child is leaf node, right child id, if right child is leaf node
                splitnodes[k][IndexDict[i][0]] = SplitNode(X_train_.columns.get_loc(features[i]), truncate(threshold[i], 4), IndexDict[children_left[i]][0], IndexDict[children_left[i]][1], IndexDict[children_right[i]][0], IndexDict[children_right[i]][1])
                
        ### write all the split rules
        counter = -1
        for i in leaf_nodes:
            counter = counter + 1
            path = []
            leafnodes[k][counter] = LeafNode(truncate(value[i][0][0], 4)) 
            leafnodes[k][counter].lb = LB.copy()
            leafnodes[k][counter].ub = UB.copy()
            
            if  parents[i][1] == 0: ### X[5] <= 0.2125
                leafnodes[k][counter].splitrule_sign.append(0)
                leafnodes[k][counter].splitnode_ids.append(split.index(int(parents[i][0])))
                # update ub of participated feature in the split node
                split_var_id = X_train_.columns.get_loc(features[int(parents[i][0])])
                split_var_criterion = truncate(threshold[int(parents[i][0])], 4)
                leafnodes[k][counter].ub[split_var_id] = split_var_criterion
                
            if parents[i][1] == 1: ### X[5] > 0.2126
                leafnodes[k][counter].splitrule_sign.append(1)
                leafnodes[k][counter].splitnode_ids.append(split.index(int(parents[i][0])))
                # update lb of participated feature in the split node
                split_var_id = X_train_.columns.get_loc(features[int(parents[i][0])])
                split_var_criterion = truncate(threshold[int(parents[i][0])], 4)
                leafnodes[k][counter].lb[split_var_id] = split_var_criterion       
    
            for j in get_path(parents, i, path):
                # path1 = []
                # temp = get_path(parents, j, path1)          
    
                if  parents[j][1] == 0:
                    leafnodes[k][counter].splitrule_sign.append(0)
                    leafnodes[k][counter].splitnode_ids.append(split.index(int(parents[j][0])))
                    # update ub of participated feature in the split node
                    split_var_id = X_train_.columns.get_loc(features[int(parents[j][0])])
                    split_var_criterion = truncate(threshold[int(parents[j][0])], 4)
                    if split_var_criterion < leafnodes[k][counter].ub[split_var_id]:
                        leafnodes[k][counter].ub[split_var_id] = split_var_criterion
                if  parents[j][1]  == 1:   
                    leafnodes[k][counter].splitrule_sign.append(1)
                    leafnodes[k][counter].splitnode_ids.append(split.index(int(parents[j][0])))
                    # update lb of participated feature in the split node
                    split_var_id = X_train_.columns.get_loc(features[int(parents[j][0])])
                    split_var_criterion = truncate(threshold[int(parents[j][0])], 4)
                    if split_var_criterion > leafnodes[k][counter].lb[split_var_id]:
                        leafnodes[k][counter].lb[split_var_id] = split_var_criterion
         
    rf_info = {'LB': LB, 'UB': UB, 'trees': trees, 'leafnodes': leafnodes, 'splitnodes': splitnodes, 'train_r_sq': r_squared_train, 'test_r_sq': r_squared_test, 'model': rf, 'params': rf_grid.best_params_}
    
    return rf_info 
#%%


def lr_info_reader(X_train, y_train, X_test, y_test):
    """
    retreive information from a pre-trained linear regression model
    """
    n_dimensions = np.shape(X_train)[1]
    LB = np.zeros(n_dimensions) 
    UB = np.ones(n_dimensions) 
        
    model = LinearRegression()
    reg = model.fit(X_train, y_train)
    print ('LR_r-squared_training: {:.2f}, LR_r-squared_testing: {:.2f}' .format(reg.score(X_train, y_train), reg.score(X_test, y_test)))
    
    coef = model.coef_
    intercept = model.intercept_
    
    lr_info = {'LB': LB, 'UB': UB, 'coef': coef, 'intercept': intercept, 'train_r_sq': reg.score(X_train, y_train), 'test_r_sq': reg.score(X_test, y_test), 'model': reg}
    return lr_info
