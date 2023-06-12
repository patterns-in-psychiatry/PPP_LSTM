#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:13:52 2022

@author: seykia
"""


#################################### IMPORTS ##################################


import warnings
warnings.filterwarnings("ignore")
from Datasets.optimize import read_dynamic_features, feature_extraction
from Datasets.optimize import compute_output_labels, read_sites
from Datasets.optimize import data_simulation, read_static_features
from Models.LSTM import prepare_network_inputs, PPP_Network, evaluate_classification
from Models.FLDM import decision_making
from Plotting.plots import plot_results
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import pickle
import os
import json
from numpy.random import seed
from tensorflow.random import set_seed



################################### USER CONFIGS ##############################


scenario = 'S5' # S1, S2, S3, S4, S5, S6

data_path = '/home/preclineu/seykia/Data/Optimize/Data/'
save_path = '/home/preclineu/seykia/Data/Optimize/Results/' + scenario + '/'


# Experiment configs
repetitions = 20 # number of experimental runs

# pretraining configs
simulation_sample_num = 10000


############################### INTERNAL CONFIGS ##############################


if scenario in ['S1','S2']:
    visits = ['2','3','5']
elif scenario in ['S3','S4','S5','S6']:
    visits = ['2','3','5','6','8']
    
if scenario in ['S1','S3']:
    past_timepoints= 1
elif scenario in ['S2','S4']:
    past_timepoints= 2
elif scenario in ['S5']:
    past_timepoints= 3
elif scenario in ['S6']:
    past_timepoints= 4

dynamic_feature_types = ['PANNS', 'PSP', 'CGI']
outcomes = ['PANNS', 'PSP', 'CGI']
    
with open('Datasets/static_features.json','r') as file_object:  
    static_feature_types = json.load(file_object)

future_timepoints = len(visits) - past_timepoints

if not os.path.isdir(save_path):
    os.mkdir(save_path)


######################### READING and PREPARING DATA ##########################
 
          
static_features = read_static_features(data_path, static_feature_types)
   
dynamic_features = read_dynamic_features(data_path, visits, dynamic_feature_types)

features, static_feature_types = feature_extraction(dynamic_features, 
                                                    static_features=static_features, 
                                            static_feature_types=static_feature_types, 
                                            aggregation='union')

remissions = compute_output_labels(dynamic_features, labels=outcomes)

subjects =  np.array(list(pd.concat([features[i][list(features[i].keys())[0]] for i in features.keys()], axis=1).dropna().index))

all_sites = read_sites(data_path)
sites = all_sites.loc[subjects]
unique_sites = list(np.unique(all_sites.loc[subjects]['site_id']))


################################### Experiment Runs ###########################


for run in range(repetitions):
    
    save_path_run = save_path + '/Run_' + str(run) + '/'
    if not os.path.isdir(save_path_run):
        os.mkdir(save_path_run)


    ################################## Data Simulation ############################
    
    rng_seed = np.random.randint(10000)    
    seed(rng_seed)
    set_seed(rng_seed)
    
    simulated_features, simulated_remissions = data_simulation(simulation_sample_num, 
                                                              dynamic_features, outcomes, static_features=static_features, 
                                                              static_feature_types=static_feature_types)
    
    X_train_sim, Y_train_sim, R_train_sim, X_static_tr_sim, X_test_sim, Y_test_sim, \
    R_test_sim, X_static_ts_sim, training_subs_sim, testing_subs_sim = \
            prepare_network_inputs(simulated_features, visits, simulated_remissions,
                                static_features, assessments=dynamic_feature_types,
                                fixed_features=True, intervals=None,
                                training_ratio=1-10/simulation_sample_num)
      
       
    ################################## MODELLING ##################################
    
        
    models = list()
    
    key = list(dynamic_features.keys())[0]
        
    R_pred = pd.DataFrame(index=subjects, columns=outcomes)
    R_pred_min = pd.DataFrame(index=subjects, columns=outcomes)
    R_pred_max = pd.DataFrame(index=subjects, columns=outcomes)
    model_ids = pd.DataFrame(index=subjects, columns=[0])
    
    
    r = 0   
    for site in unique_sites:
        testing_subs = np.array(sites.loc[sites['site_id']==site,:].index)
        training_subs = np.array(sites.loc[sites['site_id']!=site,:].index)
        
        X_train, Y_train, R_train, X_static_tr, X_test, Y_test, R_test, X_static_ts, training_subs, testing_subs = \
        prepare_network_inputs(features, visits, remissions, static_features,
                            assessments=dynamic_feature_types, fixed_features=True,
                            training_subs=training_subs, testing_subs=testing_subs)
        
        pre_trained_model = PPP_Network(X_train, X_static=X_static_tr, 
                                         Y_classes=R_train, 
                                         lstm_neurons=[60,10,4], 
                                         fc_dynamic_neurons=[30, 5, 2],
                                         fc_static_neurons=None,
                                         fc_classification_neurons=5,
                                         dropout_prob=0.1, l2_w=0.001)
        
        pre_trained_model.fit(X_train_sim, Y_train_sim, R_train_sim, X_static_tr=X_static_tr_sim, 
                          epoch_num=2, batch_size=25)
        
        models.append(pre_trained_model)
        
        models[r].fit(X_train, Y_train, R_train, X_static_tr=X_static_tr, epoch_num=50, 
                      batch_size=2)
        
        models[r].calibrate(X_train, Y_train, R_train, X_static_tr=X_static_tr)
        
        predictions, predictions_std, predicted_remissions, predicted_remissions_std = models[r].predict(X_test[-1], 
                                                                                                 X_static_ts=X_static_ts[-1],
                                                                                                  past=past_timepoints, 
                                                                                                  future=future_timepoints)
        for c in outcomes:
            R_pred.loc[testing_subs,c] = predicted_remissions[c+'_remission'][:,-1,1]
            R_pred_min.loc[testing_subs,c] = predicted_remissions_std[c+'_remission'][:,-1,1,0]
            R_pred_max.loc[testing_subs,c] = predicted_remissions_std[c+'_remission'][:,-1,1,1]
            
        model_ids.loc[testing_subs] = r
        r += 1
    
    l = []
    for c in outcomes:
        l += [c+a for a in ['_Label', '_Prob','_LB','_UB','_Recom']]
    prediction_summary = pd.DataFrame(index=subjects, columns=l + ['Model_ID'] )
    
    for c in outcomes:
        prediction_summary.loc[:,c+'_Label'] = remissions[visits[-1]][c+'_remission'].loc[subjects].to_numpy()
        prediction_summary.loc[:,c+'_Prob'] = R_pred.loc[:,c].to_numpy()
        prediction_summary.loc[:,c+'_LB'] = R_pred_min.loc[:,c].to_numpy()
        prediction_summary.loc[:,c+'_UB'] = R_pred_max.loc[:,c].to_numpy()
    prediction_summary.loc[:,'Model_ID'] = model_ids.iloc[:,0].to_numpy()
    
    prediction_summary, acc = decision_making(prediction_summary, method='fuzzy',
                                              criteria=outcomes)

        
    ################################## EVALUATION #################################
    
    
    results_original = dict()
    results_modified = dict()
    
    for c in outcomes:
        results_original[c] = evaluate_classification(remissions[visits[-1]][c+'_remission'].loc[subjects].to_numpy(np.float), 
                                      prediction_summary[c+'_Prob'].to_numpy(np.float))
        results_modified[c] = evaluate_classification(remissions[visits[-1]][c+'_remission'].loc[subjects].to_numpy(np.float), 
                                      prediction_summary[c+'_FDM'].to_numpy(np.float))
        
        
    ################################ SAVING RESULTS ###############################
     
     
    with open(save_path_run + scenario + '.pkl', 'wb') as file:
       pickle.dump({'results_original':results_original, 'results_modified':results_modified,
                    'acc':acc, 'prediction_summary':prediction_summary, 
                    'static_feature_types':static_feature_types, 'visits':visits,
                    'assessments':dynamic_feature_types, 'outcomes':outcomes}, file, protocol=4)    
   
    
############################### SUMMARIZING RESULTS ###########################
with open(save_path + '/Run_0/'+ scenario + '.pkl', 'rb') as file:
    data = pickle.load(file)

auc = {'original':{key:[] for key in data['outcomes']}, 'modified':{key:[] for key in data['outcomes']}}
bac = {'original':{key:[] for key in data['outcomes']}, 'modified':{key:[] for key in data['outcomes']}}
sen = {'original':{key:[] for key in data['outcomes']}, 'modified':{key:[] for key in data['outcomes']}}
spc = {'original':{key:[] for key in data['outcomes']}, 'modified':{key:[] for key in data['outcomes']}}
acc = {key:[] for key in data['acc'].keys()}
crt = []

for run in range(repetitions):
    
    save_path_run = save_path + '/Run_' + str(run) + '/'
    with open(save_path_run + scenario + '.pkl', 'rb') as file:
        data = pickle.load(file)
    
    for mode in ['original', 'modified']:
        for c in data['outcomes']:
            auc[mode][c].append(data['results_' + mode][c]['auc'])
            bac[mode][c].append(data['results_' + mode][c]['bac'])
            sen[mode][c].append(data['results_' + mode][c]['sensitivity'])
            spc[mode][c].append(data['results_' + mode][c]['specificity'])
      
    for key in data['acc'].keys():
        acc[key].append(data['acc'][key])
        
    crt.append((np.sum(data['prediction_summary']['PANNS_Recom'] == 'DR') + \
        np.sum(data['prediction_summary']['PANNS_Recom'] == 'PR') + \
        np.sum(data['prediction_summary']['PANNS_Recom'] == 'DN') + \
        np.sum(data['prediction_summary']['PANNS_Recom'] == 'PN')) / data['prediction_summary'].shape[0])
        
with open(save_path + scenario + '.pkl', 'wb') as file:
    pickle.dump({'auc':auc, 'bac':bac, 'sen':sen, 'spc':spc, 'acc':acc, 'crt':crt}, 
                file, protocol=4)
    
    
##############################################################################
