#!/usr/bin/env python
# coding: utf-8


#%%
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import tree
import graphviz
import sys

import datetime
import os
import keyboard

from RF_Info_Class import Tree, LeafNode, SplitNode




#%%
import math
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

#%%
    
def isolation_forest_info_reader(X_train, number_of_isolation_trees):
    """
    fit a isolation forest model, and extract RF leafnodes and splitnode info
    """
    
    # fit isolation forest
    clf = IsolationForest(n_estimators = number_of_isolation_trees, max_samples = 100, random_state = 0)
    clf.fit(X_train)
       
    # extract random forest information
    # list of LB & UB for features, 1-D list
    n_dimensions = np.shape(X_train)[1]
    LB = np.zeros(n_dimensions) 
    UB = np.ones(n_dimensions)         
    X_train = pd.DataFrame(X_train)
    # trees: leafnode_count & splitnode_count for each tree, 1-D list
    trees = {}
    # splitnodes, leafnodes: 2-D list
    splitnodes = {}
    leafnodes = {}
    
    for k in range(len(clf)):
        
        # splitnodes, leafnodes: 2-D list    
        leafnodes[k] = {}
        splitnodes[k] = {}
        
        estimator = clf.estimators_[k]
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
        features  = [X_train.columns[i] for i in estimator.tree_.feature]
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
                splitnodes[k][IndexDict[i][0]] = SplitNode(X_train.columns.get_loc(features[i]), truncate(threshold[i], 4), IndexDict[children_left[i]][0], IndexDict[children_left[i]][1], IndexDict[children_right[i]][0], IndexDict[children_right[i]][1])
                
        ### write all the split rules for splitnodes
        stack = [0]  # start with the root node id (0)
        splitnodes[k][0].splitnode_ids = []
        splitnodes[k][0].splitrule_signs = []
        while len(stack) > 0:
          i = stack.pop()
          if splitnodes[k][i].leftchild_is_leaf == 0:
            i_temp = splitnodes[k][i].leftchild_id
            splitnodes[k][i_temp].splitnode_ids = splitnodes[k][i].splitnode_ids.copy()
            splitnodes[k][i_temp].splitrule_signs = splitnodes[k][i].splitrule_signs.copy()
            splitnodes[k][i_temp].splitnode_ids.append(i)
            splitnodes[k][i_temp].splitrule_signs.append(0)
            stack.append(i_temp)
            
          if splitnodes[k][i].rightchild_is_leaf == 0:
            i_temp = splitnodes[k][i].rightchild_id
            splitnodes[k][i_temp].splitnode_ids = splitnodes[k][i].splitnode_ids.copy()
            splitnodes[k][i_temp].splitrule_signs = splitnodes[k][i].splitrule_signs.copy()
            splitnodes[k][i_temp].splitnode_ids.append(i)
            splitnodes[k][i_temp].splitrule_signs.append(1)
            stack.append(i_temp)
        
       
        ### write all the split rules for leafnodes
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
                split_var_id = X_train.columns.get_loc(features[int(parents[i][0])])
                split_var_criterion = truncate(threshold[int(parents[i][0])], 4)
                leafnodes[k][counter].ub[split_var_id] = split_var_criterion
                
            if parents[i][1] == 1: ### X[5] > 0.2126
                leafnodes[k][counter].splitrule_sign.append(1)
                leafnodes[k][counter].splitnode_ids.append(split.index(int(parents[i][0])))
                # update lb of participated feature in the split node
                split_var_id = X_train.columns.get_loc(features[int(parents[i][0])])
                split_var_criterion = truncate(threshold[int(parents[i][0])], 4)
                leafnodes[k][counter].lb[split_var_id] = split_var_criterion       
    
            for j in get_path(parents, i, path):
                # path1 = []
                # temp = get_path(parents, j, path1)          
    
                if  parents[j][1] == 0:
                    leafnodes[k][counter].splitrule_sign.append(0)
                    leafnodes[k][counter].splitnode_ids.append(split.index(int(parents[j][0])))
                    # update ub of participated feature in the split node
                    split_var_id = X_train.columns.get_loc(features[int(parents[j][0])])
                    split_var_criterion = truncate(threshold[int(parents[j][0])], 4)
                    if split_var_criterion < leafnodes[k][counter].ub[split_var_id]:
                        leafnodes[k][counter].ub[split_var_id] = split_var_criterion
                if  parents[j][1]  == 1:   
                    leafnodes[k][counter].splitrule_sign.append(1)
                    leafnodes[k][counter].splitnode_ids.append(split.index(int(parents[j][0])))
                    # update lb of participated feature in the split node
                    split_var_id = X_train.columns.get_loc(features[int(parents[j][0])])
                    split_var_criterion = truncate(threshold[int(parents[j][0])], 4)
                    if split_var_criterion > leafnodes[k][counter].lb[split_var_id]:
                        leafnodes[k][counter].lb[split_var_id] = split_var_criterion
         
    isolation_forest_info = {'trees': trees, 'leafnodes': leafnodes, 'splitnodes': splitnodes, 'model': clf}
    
    return isolation_forest_info    

