#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:25:19 2020

@author: seykia
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from Datasets.optimize import compute_classifier_inputs


def classifier_evaluation(features, remissions, cv=10, repetitions=10, assessments=['PANNS'],
                          static_feature_types=None, criterion='PANNS',
                          feature_visits=['2'], label_visit='5', missing_values='impute',
                          classifiers=None, classifier_names=None):

    if classifiers is None:
        classifiers = [
            LogisticRegression(),
            SVC(kernel="linear", C=1),
            SVC(gamma=2, C=1),
            KNeighborsClassifier(3),
            MLPClassifier(hidden_layer_sizes=25, alpha=1),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            GaussianNB(),
            GaussianProcessClassifier(1.0 * RBF(1.0)), 
            ]
        classifier_names = ['LR', "LSVM", "RBF-SVM", "KNN", "MLP", "DT", "RF", "AdaBoost", 'GB',
         "NB", "GP"]

    aucs = np.zeros([repetitions, len(classifiers)])    
        
    X, Y, subjects = compute_classifier_inputs(features, remissions, assessments,
                          criterion, static_feature_types, feature_visits, 
                          label_visit, missing_values)

    i = 0
    for name, clf in zip(classifier_names, classifiers):
        for r in range(repetitions):     
            scaler = StandardScaler()
            pipeline = Pipeline([('transformer', scaler), ('estimator', clf)])
            rand_idx = np.random.permutation(X.shape[0])
            X = X[rand_idx,:]
            Y = Y[rand_idx,:]
            scores = cross_val_score(pipeline, X, Y.squeeze(), cv=cv, scoring='roc_auc')
            aucs[r, i] = scores.mean()
        print(name + " Classifier: AUC= %f +/- %f" %(np.mean(aucs[:,i]), 
                                                     np.std(aucs[:,i])))
        i += 1
    
    return aucs