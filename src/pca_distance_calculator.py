#!/usr/bin/env python
# coding: utf-8

#%%

import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler  
standard_scaler = StandardScaler()
from sklearn.decomposition import PCA

#%%

def PCA_Distance_SinglePoint(x, data):
    """
    Compute the distance between x and a reduced space of data  
    x    : vector or matrix of data with, say, p columns.
    data : x_train
    """
    cov = np.cov(data.T)
    std_dev = np.sqrt(np.diag(cov))
    # print(std_dev)
    # print(np.mean(data, axis = 0))
    x_minus_mu = x - np.mean(data, axis = 0) # sol stores the decision variable
    # pca = PCA(0.9) # pca = PCA(0.9)
    pca = PCA(n_components=data.shape[1]-1) # data.shape[1]
    data = standard_scaler.fit_transform(data)
    # print(np.mean(data, axis = 0))
    pca.fit(data) # fit PCA on training set
    # pca.components_ # shape is k*n
    # print(pca.components_)
    # print(pca.explained_variance_ratio_)
    term_1 = np.identity(data.shape[1]) - np.dot(pca.components_.T, pca.components_)
    term_12 = np.dot(term_1, sp.linalg.inv(np.diag(std_dev)))
    term_123 = np.dot(term_12, x_minus_mu)
    pca_distance = np.dot(term_123, term_123.T) # dot product of a vector and itself: a^2+b^2+...
    # print(np.dot(pca.components_.T, pca.components_))
    # term_1 = np.dot(pca.components_.T, sp.linalg.inv(np.diag(pca.explained_variance_)))
    # term_12 = np.dot(term_1, pca.components_)
    # term_123 = np.dot(term_12, x_minus_mu)
    # pca_distance = np.dot(term_123, x_minus_mu.T)     
    
    return pca_distance 


#%%

def PCA_Orthogonal_Distance_cutoff_value(data, alpha):
    """
    Calculate the cutoff value of the orthogonal distances of the sample points (Section 4.5 - PCA)
    data: x_train
    """
    pca = PCA(n_components=data.shape[1]-1) # data.shape[1]
    data = standard_scaler.fit_transform(data)
    pca.fit(data) # fit PCA on training set
    
    P_pk = pca.components_.T # shape is p*k, k is number of components, p is number of independent variables (original dimension)
    p = data.shape[1] # number of independent variables (original dimension)
    n = data.shape[0] # n is number of points in data
    mu_hat = np.mean(data, axis = 0)
    T_nk_first_component = (data - np.matmul(np.ones((n, 1)), mu_hat.reshape(1, p))) # n is number of points in data   
    T_nk = np.matmul(T_nk_first_component, P_pk) # k is snumber of dimensions in subspace 
    Orthogonal_Distance_allpoints = []
    for i in range(len(data)):
        point_tmp = data[i]
        OD_point_tmp_vector = point_tmp - mu_hat - np.matmul(P_pk, T_nk[i]).reshape(1, p)
        OD_point_tmp = np.dot(OD_point_tmp_vector, OD_point_tmp_vector.T)
        Orthogonal_Distance_allpoints.append(OD_point_tmp[0][0])
    
    mu_hat_OD = np.mean(Orthogonal_Distance_allpoints, axis = 0)
    cov_hat_OD = np.std(Orthogonal_Distance_allpoints)
    cutoff_value = np.power(mu_hat_OD + cov_hat_OD * sp.stats.norm.ppf(1-alpha), 3/2)
    
    return cutoff_value 