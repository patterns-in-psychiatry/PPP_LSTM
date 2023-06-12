#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:47:44 2021

@author: seykia
"""

import numpy as np
import copy
import itertools

def robust_minmax(array):
    
    shape = list(array.shape[:-1])
    last_dim = len(array.shape) - 1 
    
    array = np.reshape(array,[np.prod(array.shape[0:last_dim]), array.shape[last_dim]])
    
    min_array = np.zeros(array.shape[0])
    max_array = np.zeros(array.shape[0])
    
    n = int(np.round(array.shape[1] / 10))
    
    idx = array.argsort(axis=1)[:,-n:]
    for i in range(array.shape[0]):
        max_array[i] = np.median(array[i, idx[i,:]])
        
    idx = array.argsort(axis=1)[:,0:n]
    for i in range(array.shape[0]):
        min_array[i] = np.median(array[i, idx[i,:]])
    
    max_array = np.reshape(max_array, shape+[1])
    min_array = np.reshape(min_array, shape+[1])
    minmax = np.concatenate((min_array, max_array), axis=last_dim)

    return minmax


def augment_data(X_train, Y_train):
    
    modalities_num = len(X_train.keys())
    combs = []
    for i in range(1,modalities_num-1):	
        combs = combs + list(itertools.combinations(range(modalities_num), i))
        
    keys = list(X_train.keys())
    augmeted_X = copy.deepcopy(X_train)
    augmeted_Y = copy.deepcopy(Y_train)
    for i in range(len(combs)):
        temp = copy.deepcopy(X_train)
        k = [keys[j] for j in combs[i]]
        for j in k:
            temp[j][:,:] = np.nan
        for key in X_train.keys():
            augmeted_X[key] = np.concatenate([augmeted_X[key],temp[key]], axis=0)
        augmeted_Y = np.concatenate([augmeted_Y, Y_train], axis=0)
        if i%100==0:
            print(i)
    
    return augmeted_X, augmeted_Y
