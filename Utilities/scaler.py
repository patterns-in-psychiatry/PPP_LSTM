#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:33:40 2021

@author: seykia
"""


import numpy as np


class scaler:
    
    def __init__(self, scaler_type='standardize', tail=0.01):
        
        self.scaler_type = scaler_type
        self.tail = tail
        
        if self.scaler_type not in ['standardize', 'minmax', 'robminmax']:
             raise ValueError("Undifined scaler type!")  
        
        
    def fit(self, X):
        
        if self.scaler_type == 'standardize':
            
            self.m = np.nanmean(X, axis=0)
            self.s = np.nanstd(X, axis=0)
            
        elif self.scaler_type == 'minmax':
            self.min = np.nanmin(X, axis=0)
            self.max = np.nanmax(X, axis=0)
        
        elif self.scaler_type == 'robminmax':
            self.min = np.zeros([X.shape[1],])
            self.max = np.zeros([X.shape[1],])
            for i in range(X.shape[1]):
                mask = np.isnan(X[:,i])==False
                masked_X = X[mask,i]
                self.min[i] = np.nanmedian(np.sort(masked_X)[0:int(np.round(masked_X.shape[0] * self.tail))])
                self.max[i] = np.nanmedian(np.sort(masked_X)[-int(np.round(masked_X.shape[0] * self.tail)):])   
                
                
    def transform(self, X, adjust_outliers=True):
        
        if self.scaler_type == 'standardize':
            
            X = (X - self.m) / self.s 
        
        elif self.scaler_type in ['minmax', 'robminmax']:
            
            X = (X - self.min) / (self.max - self.min)
            
            if adjust_outliers:
                
                X[X < 0] = 0
                X[X > 1] = 1
            
        return X
    
    def inverse_transform(self, X, index=None):
        
        if self.scaler_type == 'standardize':
            if index is None:
                X = X * self.s + self.m
            else:
                X = X * self.s[index] + self.m[index]
        
        elif self.scaler_type in ['minmax', 'robminmax']:
            if index is None:
                X = X * (self.max - self.min) + self.min 
            else:
                X = X * (self.max[index] - self.min[index]) + self.min[index]
        return X
    
    def fit_transform(self, X, adjust_outliers=True):
        
        if self.scaler_type == 'standardize':
            
            self.m = np.nanmean(X, axis=0)
            self.s = np.nanstd(X, axis=0)
            X = (X - self.m) / self.s 
            
        elif self.scaler_type == 'minmax':
            
            self.min = np.nanmin(X, axis=0)
            self.max = np.nanmax(X, axis=0)
            X = (X - self.min) / (self.max - self.min)
        
        elif self.scaler_type == 'robminmax':  
            self.min = np.zeros([X.shape[1],])
            self.max = np.zeros([X.shape[1],])
            for i in range(X.shape[1]):
                mask = np.isnan(X[:,i])==False
                masked_X = X[mask,i]
                self.min[i] = np.nanmedian(np.sort(masked_X)[0:int(np.round(masked_X.shape[0] * self.tail))])
                self.max[i] = np.nanmedian(np.sort(masked_X)[-int(np.round(masked_X.shape[0] * self.tail)):])   
                
            X = (X - self.min) / (self.max - self.min)
            
            if adjust_outliers:             
                X[X < 0] = 0
                X[X > 1] = 1
        
        return X