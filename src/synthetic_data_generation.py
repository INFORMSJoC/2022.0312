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

import gurobipy as gp
from gurobipy import GRB

from pyDOE import *
from skopt.space import Space
from skopt.sampler import Lhs

import sklearn.datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
min_max_scaler = MinMaxScaler()

#%%

def SimulatedDataset(lambda_std_noise, random_state, true_function, n_dimensions, n_samples, mu):
    """
    simulate dataset by multi normal distributed 
    """
    matrixSize = n_dimensions
    Cov = sklearn.datasets.make_spd_matrix(matrixSize, random_state = random_state)
    n = n_samples

    np.random.seed(random_state)
    X = np.random.multivariate_normal(mu, Cov, n)
   
    
    y_expectation = np.apply_along_axis(true_function, 1, X)
    std_noise = lambda_std_noise * np.std(y_expectation)
    y_observed = np.apply_along_axis(true_function, 1, X) + np.random.normal(0, std_noise, n_samples)
        
    df_1 = pd.DataFrame(X)
    df_2 = pd.DataFrame(y_observed, columns = ['y_observed'])
    df_3 = pd.DataFrame(y_expectation, columns = ['y_expectation'])
    df = pd.concat([df_1, df_2, df_3], axis=1, sort=False)
    BestSampleWithNoise = df['y_observed'].min() 
    CorrExpectation_BestSample = df.loc[df['y_observed'].idxmin()]['y_expectation']
    
    return X, y_observed, y_expectation, BestSampleWithNoise, CorrExpectation_BestSample


#%%

def PreprocessData(x_simulated, y_simulated, true_function):
    """
    mix max x to [0, 1], standardize y;
    split training and test sets     
    """
    standard_scaler_y = StandardScaler()
    X_scaled = min_max_scaler.fit_transform(x_simulated)
    y_scaled = standard_scaler_y.fit_transform(y_simulated.reshape(-1,1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.3, random_state = 1)
    
    X = min_max_scaler.inverse_transform(X_train)
    df_1 = pd.DataFrame(X)
    df_2 = pd.DataFrame(y_train, columns = ['y_observed'])
    y_expectation = np.apply_along_axis(true_function, 1, X)
    df_3 = pd.DataFrame(y_expectation, columns = ['y_expectation'])
    df = pd.concat([df_1, df_2, df_3], axis=1, sort=False)
    BestSampleWithNoise = df['y_observed'].min() 
    CorrExpectation_BestSample = df.loc[df['y_observed'].idxmin()]['y_expectation']
   
    return X_scaled, y_scaled, X_train, X_test, y_train, y_test, min_max_scaler, standard_scaler_y, CorrExpectation_BestSample


