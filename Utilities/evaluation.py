#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:02:35 2021

@author: seykia
"""

import numpy as np
from sklearn.calibration import calibration_curve



def compute_MSLL(ytrue, ypred, ypred_var, train_mean = None, train_var = None): 
    """ Computes the MSLL or MLL (not standardized) if 'train_mean' and 'train_var' are None.
    
        Basic usage::
            
            MSLL = compute_MSLL(ytrue, ypred, ytrue_sig, noise_variance, train_mean, train_var)
        
        where
        
        :param ytrue     : n*p matrix of true values where n is the number of samples 
                           and p is the number of features. 
        :param ypred     : n*p matrix of predicted values where n is the number of samples 
                           and p is the number of features. 
        :param ypred_var : n*p matrix of summed noise variances and prediction variances where n is the number of samples 
                           and p is the number of features.
            
        :param train_mean: p dimensional vector of mean values of the training data for each feature.
        
        :param train_var : p dimensional vector of covariances of the training data for each feature.

        :returns loss    : p dimensional vector of MSLL or MLL for each feature.

    """
    
    if train_mean is not None and train_var is not None: 
            
        # compute MSLL:
        loss = np.mean(0.5 * np.log(2 * np.pi * ypred_var) + (ytrue - ypred)**2 / (2 * ypred_var) - 
                       0.5 * np.log(2 * np.pi * train_var) - (ytrue - train_mean)**2 / (2 * train_var), axis = 0)
        
    else:   
        # compute MLL:
        loss = np.mean(0.5 * np.log(2 * np.pi * ypred_var) + (ytrue - ypred)**2 / (2 * ypred_var), axis = 0)
        
    return loss


def calibration_error(y_true, y_pred, n_bins=10):
    
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.searchsorted(bins[1:-1], y_pred)
        bin_total = np.bincount(binids, minlength=len(prob_true))
        nonzero = bin_total != 0
        ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
    except:
        ece = calibration_error(y_true, y_pred, n_bins=n_bins-1)
        
    return ece