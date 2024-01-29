#!/usr/bin/env python
# coding: utf-8

#%%
import numpy as np
import scipy as sp

#%%

def MDistance_Squared_SinglePoint(x = None, data = None, cov = None):
    """
    Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data, axis = 0)
    if cov is None:
        cov = np.cov(data.T)
    else:
        cov = cov
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    MDist = np.dot(left_term, x_minus_mu.T)
    return MDist # for a single data point


def MDistance_Squared_MultiplePoints(x = None, data = None, mean = None, cov = None):
    """
    Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    if mean is None:
        mean = np.array(np.mean(data, axis = 0))
    x_minus_mu = x - mean
    if cov is None:
        cov = np.cov(data.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal() # for multiple data points

#%%
