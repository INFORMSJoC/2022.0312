#!/usr/bin/env python
# coding: utf-8

#%%
import numpy as np

#%%

# global optimization benchmark functions
def ackley(x):
    return -20*np.exp(-0.2*np.sqrt(0.5*(np.square(x[0]) + np.square(x[1])))) - np.exp(0.5 * ( np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.exp(1) + 20
def adjiman(x): 
    return np.cos(x[0]) * np.sin(x[1]) - x[0] /(np.square(x[1]) + 1)
def alpine_1(x):
    Sum = 0
    for i in range(6):
        Sum = Sum + abs(x[i] * np.sin(x[i]) + 0.1 * x[i])
    return Sum 
def alpine_2(x):
    Product = 1
    for i in range(6):
        Product = Product * np.sqrt(x[i]) * np.sin(x[i])
    return Product 

def beale(x):
    return np.square(1.5 - x[0] + x[0]*x[1]) + np.square(2.25 - x[0] + x[0]*np.square(x[1])) + np.square(2.625 - x[0] + x[0]*np.power(x[1], 3))

def chung(x):
    Sum = 0
    for i in range(6):
        Sum = Sum + np.square(x[i])
    return np.square(Sum)

def easom(x): 
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-np.square(x[0]-np.pi)-np.square(x[1]-np.pi))
def eggcrate(x): 
    return np.square(x[0]) + np.square(x[1]) + 25 * (np.square(np.sin(x[0])) + np.square(np.sin(x[1])))
def eggholder(x): 
    return - x[0] * np.sin(np.sqrt(np.abs(x[0] - x[1] - 47))) - (x[1] + 47) * np.sin(np.sqrt(np.abs(1/2 * x[0] + x[1] + 47)))
def exponential(x): 
    return -np.exp(-0.5 * (np.square(x[0]) + np.square(x[1])))

def griewank(x):
    Sum = 0
    Product = 1
    for i in range(4):
        Sum = Sum + np.square(x[i])/4000
        Product = Product * np.cos(x[i] / np.sqrt(i+1))
    return Sum - Product + 1

def hosaki(x): 
    return (1 - 8*x[0] + 7*np.square(x[0]) - 7/3*np.power(x[0], 3) + 1/4*np.power(x[0], 4)) * np.square(x[1]) * np.exp(-x[0])

def mishra_11(x):
    Sum = 0
    Product = 1
    for i in range(8):
        Sum = Sum + np.abs(x[i])
        Product = Product * np.abs(x[i])
    return np.square(Sum/8 - np.power(Product, 1/8))

def paviani(x):
    Sum = 0
    Product = 1
    for i in range(10):
        Sum = Sum + np.square(np.log(10 - x[i])) + np.square(np.log(x[i] - 2))
        Product = Product * x[i]
    return Sum - np.power(Product, 0.2)
def peaks(x):
    return 3*np.square(1-x[0])*np.exp(-np.square(x[0])-np.square(x[1]+1))-10*(x[0]/5-np.power(x[0],3)-np.power(x[1],5))*np.exp(-np.square(x[0])-np.square(x[1]))-1/3*np.exp(-np.square(x[0]+1)-np.square(x[1])) 

def peaks_adj(x):
    return 6.55 + 3*np.square(1-x[0])*np.exp(-np.square(x[0])-np.square(x[1]+1))-10*(x[0]/5-np.power(x[0],3)-np.power(x[1],5))*np.exp(-np.square(x[0])-np.square(x[1]))-1/3*np.exp(-np.square(x[0]+1)-np.square(x[1])) 

def powell(x):
    return np.square(x[0] + 10*x[1]) + 5*np.square(x[2]-x[3]) + np.power(x[1]-2*x[2], 4) + 10*np.power(x[0]-x[3], 4)
def price_1(x):
    return np.square(abs(x[0]) - 5) + np.square(abs(x[1]) - 5)

def qing(x):
    Sum = 0
    for i in range(8):
        Sum = Sum + np.square(np.square(x[i]) - (i+1))
    return Sum 
def quintic(x):
    Sum = 0
    for i in range(5):
        Sum = Sum + abs(np.power(x[i], 5) - 3*np.power(x[i], 4) + 4*np.power(x[i], 3) + 2*np.power(x[i], 2) - 10*x[i] - 4)
    return Sum 

def rastrigin(x):
    Sum = 100
    for i in range(10):
        Sum = Sum + np.square(x[i]) - 10*np.cos(2*np.pi*x[i])
    return Sum 

def salomon(x):
    return 1- np.cos(2*np.pi*np.sqrt(np.square(x[0])+np.square(x[1]))) + 0.1*np.sqrt(np.square(x[0])+np.square(x[1]))
def salomon_4(x):
    return 1- np.cos(2*np.pi*np.sqrt(np.square(x[0])+np.square(x[1])+np.square(x[2])+np.square(x[3]))) + 0.1*np.sqrt(np.square(x[0])+np.square(x[1])+np.square(x[2])+np.square(x[3]))


def wayburn_seader_1(x):
    return np.square(np.power(x[0], 6) + np.power(x[1], 4) - 17) + np.square(2*x[0] + x[1] - 4)

