#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from pca_distance_calculator import PCA_Distance_SinglePoint
from mahalanobis_distance_calculator import MDistance_Squared_SinglePoint
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull


#%%
def evaluate_instance_SVM(instance, svm, L):
    '''
    return value 1 if the sample labeled by One Class SVM as inlier,
    i.e., value calculated from decision function >= 0 (equivalent to left side >= rho),
    smaller rho leads to more samples labeled as inlier
    '''
    sv = svm.support_vectors_.tolist()
    alpha = svm.dual_coef_.ravel().tolist()
    gamma = svm._gamma
    
    d = {}
    k = {}
    dist = 0
    for i in range(len(sv)):
        d[i] = 0
        for j in range(len(instance)):
            d[i] += (sv[i][j] - instance[j]) * (sv[i][j] - instance[j])
        k[i] = np.exp(-gamma * d[i])
        dist += alpha[i] * k[i]
    if dist < L:
        return 0
    else:
        return 1
    
#%%

def volume_SVM(X, svm, L):
    n_catch_by_SVM = 0
    n_samples = len(X)
    for i in range(len(X)):
        instance = X[i].reshape(1,-1)[0]
        n_catch_by_SVM += evaluate_instance_SVM(instance, svm, L)
    points_captured_ptg = n_catch_by_SVM / n_samples
    return points_captured_ptg


#%%

def evaluate_instance_IF(instance, clf, L):
    '''
    return value 1 if the sample labeled by Isolation Forest as inlier, 
    i.e., depth for all trees > depth parameter in PMO,
    smaller L  leads to more samples labeled as inlier
    '''
    i = 0
    while i < 100:
        tree = clf[i]
        path_tmp = tree.decision_path(instance).toarray() # ExtraTreeRegressor has function to return he decision path, 1 denotes the nodes that the sample traverses through
        depth_tmp = np.sum(path_tmp[0]) - 1
        if depth_tmp <= L:
            return 0
        else:
            i += 1
    return 1

#%%

def volume_IF(X, clf, L):
    n_catch_by_IF = 0
    n_samples = len(X)
    for i in range(len(X)):
        instance = X[i].reshape(1,-1)
        n_catch_by_IF += evaluate_instance_IF(instance,clf,L)
    points_captured_ptg = n_catch_by_IF / n_samples
    return points_captured_ptg
       
#%%
'''
    return value 1 if the L1 distance between the sample and any one of the training points is less than or equal to L 
'''
def evaluate_instance_KNN(instance, X_train, L1_per_dimension):
    i = 0
    L1_all_dimensions = L1_per_dimension * X_train.shape[1]
    while i < len(X_train):
        X_train_tmp = X_train[i]
        dist = 0
        for j in range(len(instance)):
            # print(instance[j])
            dist += np.abs(X_train_tmp[j] - instance[j])
        if dist <= L1_all_dimensions:
            return 1
        else:
            i += 1
    return 0

#%%

def volume_KNN(X, X_train, L1_per_dimension):
    n_samples = len(X)
    L1_all_dimensions = L1_per_dimension * X_train.shape[1]
    n_catch_by_KNN = 0
    for i in range(len(X)):
        instance = X[i].reshape(1,-1)[0]
        n_catch_by_KNN += evaluate_instance_KNN(instance, X_train, L1_all_dimensions)
    points_captured_ptg = n_catch_by_KNN / n_samples
    return points_captured_ptg
    
#%%
'''
    return value 1 if the Mahalanobis distance between the sample and the training data is less than or equal to L 
'''
def evaluate_instance_MD(instance, X_train, L):
    dist = MDistance_Squared_SinglePoint(instance, X_train)
    if dist <= L:
        return 1
    else:
        return 0

#%%

def volume_MD(X, X_train, L):
    n_samples = len(X)
    n_catch_by_MD = 0
    for i in range(len(X)):
        instance = X[i].reshape(1,-1)[0]
        n_catch_by_MD += evaluate_instance_MD(instance, X_train, L)
    points_captured_ptg = n_catch_by_MD / n_samples
    return points_captured_ptg
    
#%%
'''
    return value 1 if the Mahalanobis distance between the sample and the subspace of training data is less than or equal to L 
'''
def evaluate_instance_PCA(instance, X_train, L):
    dist = PCA_Distance_SinglePoint(instance, X_train)
    if dist <= L:
        return 1
    else:
        return 0

#%%

def volume_PCA(X, X_train, L):
    n_samples = len(X)
    n_catch_by_PCA = 0
    for i in range(len(X)):
        instance = X[i].reshape(1,-1)[0]
        n_catch_by_PCA += evaluate_instance_PCA(instance, X_train, L)
    points_captured_ptg = n_catch_by_PCA / n_samples
    return points_captured_ptg
      
#%%
'''
    return value 1 if the sample drops in the subspace of training data 
'''
def evaluate_instance_CH(instance, X_train):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """   
    p = instance
    hull = ConvexHull(X_train)
    tol = 1e-12
    return int(np.all(hull.equations[:,:-1] @ p.T + np.repeat(hull.equations[:,-1][None,:], len(p), axis=0).T <= tol, 0))
#%%

def volume_CH(X, X_train):
    n_samples = len(X)
    n_catch_by_CH = 0
    for i in range(len(X)):
        instance = X[i].reshape(1,-1)
        n_catch_by_CH += evaluate_instance_CH(instance, X_train)
    points_captured_ptg = n_catch_by_CH / n_samples
    return points_captured_ptg
    