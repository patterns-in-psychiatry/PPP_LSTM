#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 19:40:24 2021

@author: seykia
"""

#################################### IMPORTS ##################################


import warnings
warnings.filterwarnings("ignore")
from Datasets.optimize import read_dynamic_features, feature_extraction, swn_transform
from Datasets.optimize import compute_output_labels, create_static_features_json
from Datasets.optimize import data_simulation, read_static_features, read_prognosis
from Models.LSTM import prepare_network_inputs, PPP_Network, evaluate_classification
from Models.LSTM import evaluate_regression, prepare_TLSTM_inputs
from Utilities.interpretation import counterfactual_interpretation
from Models.FLDM import decision_making
from Plotting.plots import plot_results, plot_CFI_group_effect
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import pickle
import os
import json
from numpy.random import seed
from tensorflow.random import set_seed
from sklearn.calibration import calibration_curve, CalibrationDisplay


################################### USER CONFIGS ##############################

study = 'daniel'
base_path = '/home/preclineu/seykia/Data/Optimize/' 
data_path = base_path + 'Data/'

create_static_features_json(data_path, study)

# Here you should specify the visits you are interested in
#visits = ['2','3','4','5','6','7','8','10','12','16','20'] #'20'
#visits = ['2','3','5']
visits = ['2','3','5','6','8']
intervals = [0,1,2,1,2]

# Here specify the assessments you want to retrieve from optimize data, assessments = ['PANNS', 'PSP', 'CGI']
assessments = ['PANNS', 'PSP', 'CGI']
criteria = ['PANNS', 'PSP', 'CGI']

with_dynamics = True
with_statics = True
use_tlstm = True
use_nanLSTM = False
use_c = False
calibrate = True
with_residual = True
with_checkpoints = False
with_swn_transform = False


if not use_tlstm:
    intervals = None

#In TLSTM pls choose directory to save varibale trajectory for learn_parametrs
tlstm_decay_function = 'parametric_sigmoid' #options: 'original', 'parametric_sigmoid', 'nonparametric'

# number of past timepoints that asre used for prediction. Only used if with_dynamics.
past_timepoints= 3
    

if with_statics:
    with open(data_path + study + '_static_features.json','r') as file_object:  
        static_feature_types = json.load(file_object)
else: 
    static_feature_types = None

# Experiment configs
repetitions = 10 # number of experimental runs
pre_train = True
validation_split = 0 # if not 0, then it is used as validation portion in early stopping.
perform_cfi = False # whether to perfom counterfactual interpretation

# pretraining configs
simulation_sample_num = 10000
pre_epoch_num = 2
pre_batch_size = 25

# training configs
epoch_num = 50
batch_size = 1
fold_num = 10 # number of folds in K-fold cross-validation

# Network architecture parameters
dropout_prob = 0.1
l2_w = 0.001

if with_residual:
    fc_dynamic_neurons =  [30, 5, 2]
else:
    fc_dynamic_neurons =  [30, 5, 5]
    
lstm_neuron_num =  [50,10,5]  
fc_classification_neurons= 5
fc_static_neurons =  None #[10, 10, 5, 5, 10]

decision_making_method = 'fuzzy' # fuzzy or plain decision making.


############################### INTERNAL CONFIGS ##############################

future_timepoints = len(visits) - past_timepoints

if with_statics:
    if with_dynamics:
        experiment_name = '_'.join(assessments+['DEMO']) + '_v' + '_v'.join(visits) + '_' + str(past_timepoints)
    else:
        experiment_name = 'DEMO' + '_v' + '_v'.join(visits)
else:
    experiment_name = '_'.join(assessments) + '_v' + '_v'.join(visits) + '_' + str(past_timepoints)

save_path = base_path + 'Results/' + experiment_name + '/'

if not os.path.isdir(base_path + 'Results/'):
    os.mkdir(base_path + 'Results/')

if not os.path.isdir(save_path):
    os.mkdir(save_path)

######################### READING and PREPARING DATA ##########################
 
          
if with_statics:
    static_features = read_static_features(data_path, static_feature_types)
    if with_swn_transform:
        static_features, static_feature_types = swn_transform(static_features, 
                                                              static_feature_types)
else:
    static_features = None

dynamic_features = read_dynamic_features(data_path, visits, assessments)

features, static_feature_types = feature_extraction(dynamic_features, 
                                                    static_features=static_features, 
                                            static_feature_types=static_feature_types, 
                                            aggregation='union')

remissions = compute_output_labels(dynamic_features, labels=criteria)

subjects =  np.array(list(pd.concat([features[i][list(features[i].keys())[0]] for i in features.keys()], axis=1).dropna().index))


############################## EXPERT PERFORMANCE #############################


prognosis = read_prognosis(data_path, subjects)

expert_performance = evaluate_classification(remissions[visits[-1]]['PANNS_remission'].loc[subjects].to_numpy(), 
                                  prognosis['V1_PROGNOSIS'].loc[subjects].to_numpy())

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
                                                              dynamic_features, criteria, static_features=static_features, 
                                                              static_feature_types=static_feature_types)
    
    if with_dynamics:
        if use_tlstm:
                X_train_sim, Y_train_sim, R_train_sim, X_static_tr_sim, X_test_sim, Y_test_sim, R_test_sim,\
                    X_static_ts_sim, training_subs_sim, testing_subs_sim = \
                    prepare_TLSTM_inputs(simulated_features, visits, intervals, simulated_remissions, 
                                     static_features, assessments=assessments,
                                     fixed_features=with_statics, training_ratio=1-10/simulation_sample_num)
        else:    
            X_train_sim, Y_train_sim, R_train_sim, X_static_tr_sim, X_test_sim, Y_test_sim, \
            R_test_sim, X_static_ts_sim, training_subs_sim, testing_subs_sim = \
                    prepare_network_inputs(simulated_features, visits, intervals, simulated_remissions,
                                        static_features, assessments=assessments,
                                        fixed_features=with_statics,
                                        training_ratio=1-10/simulation_sample_num)
    else:
        X_train_sim, Y_train_sim, R_train_sim, X_static_tr_sim, X_test_sim, Y_test_sim, \
        R_test_sim, X_static_ts_sim, training_subs_sim, testing_subs_sim = \
                prepare_network_inputs(simulated_features, visits, simulated_remissions,
                                    static_features, assessments=[], 
                                    fixed_features=with_statics,
                                    training_ratio=1-10/simulation_sample_num)
      
       
    ################################## MODELLING ##################################
    
        
    models = list()
    
    key = list(dynamic_features.keys())[0]
    regression_results = dict()
    for assessment in assessments:
        regression_results[assessment] = np.zeros([2, len(subjects), future_timepoints, 
                                          features[key][assessment].shape[1]])
        
    R_pred = pd.DataFrame(index=subjects, columns=criteria)
    R_pred_min = pd.DataFrame(index=subjects, columns=criteria)
    R_pred_max = pd.DataFrame(index=subjects, columns=criteria)
    model_ids = pd.DataFrame(index=subjects, columns=[0])
    
    kf = StratifiedKFold(n_splits=fold_num, shuffle=True)
    
    r = 0   
    
    for train_index, test_index in kf.split(subjects, remissions[visits[-1]]['PANNS_remission'].loc[subjects].to_numpy()):
        
        if with_dynamics:
            if use_tlstm:
                X_train, Y_train, R_train, X_static_tr, X_test, Y_test, R_test, X_static_ts, training_subs, testing_subs = \
                prepare_TLSTM_inputs(features, visits, intervals, remissions, 
                                     static_features, assessments=assessments,
                                     fixed_features=with_statics, training_subs=subjects[train_index],
                                     testing_subs=subjects[test_index])
            else:
                X_train, Y_train, R_train, X_static_tr, X_test, Y_test, R_test, X_static_ts, training_subs, testing_subs = \
                prepare_network_inputs(features, visits, remissions, static_features, intervals=intervals,
                                    assessments=assessments, fixed_features=with_statics,
                                    training_subs=subjects[train_index], testing_subs=subjects[test_index])
                
        else:
             X_train, Y_train, R_train, X_static_tr, X_test, Y_test, R_test, X_static_ts, training_subs, testing_subs = \
                prepare_network_inputs(features, visits, remissions, static_features,
                                    assessments=[], fixed_features=with_statics,
                                    training_subs=subjects[train_index], testing_subs=subjects[test_index])
            
        pre_trained_model = PPP_Network(X_train, X_static=X_static_tr, 
                                         Y_classes=R_train, use_tlstm=use_tlstm, 
                                         with_nan=use_nanLSTM, lstm_neurons=lstm_neuron_num, 
                                         fc_dynamic_neurons=fc_dynamic_neurons,
                                         fc_static_neurons=fc_static_neurons,
                                         fc_classification_neurons=fc_classification_neurons,
                                         dropout_prob=dropout_prob, l2_w=l2_w, use_c=use_c,
                                         with_residual=with_residual,
                                         tlstm_decay_function=tlstm_decay_function)
        
    
        if pre_train:
            pre_trained_model.fit(X_train_sim, Y_train_sim, R_train_sim, X_static_tr=X_static_tr_sim, 
                          epoch_num=pre_epoch_num, batch_size=pre_batch_size)
        
        models.append(pre_trained_model)
        
        if with_checkpoints:
            checkpoint_path = save_path_run +'/kf_' + str(r) + '/' + "cp-{epoch:02d}.ckpt" 
        else:
            checkpoint_path = None
        
        models[r].fit(X_train, Y_train, R_train, X_static_tr=X_static_tr, epoch_num=epoch_num, 
                batch_size=batch_size, validation_split=validation_split, checkpoint_path=checkpoint_path)
        
        if calibrate:
            models[r].calibrate(X_train, Y_train, R_train, X_static_tr=X_static_tr)
        
        if with_dynamics:
            if with_statics:
                predictions, predictions_std, predicted_remissions, predicted_remissions_std = models[r].predict(X_test[-1], 
                                                                                                     X_static_ts=X_static_ts[-1],
                                                                                                      past=past_timepoints, 
                                                                                                      future=future_timepoints,
                                                                                                      intervals=intervals)
            else:
               predictions, predictions_std, predicted_remissions, predicted_remissions_std= models[r].predict(X_test[-1], past=past_timepoints,
                                                                                                future=future_timepoints,
                                                                                                      intervals=intervals)   
            for assessment in assessments:
                regression_results[assessment][0,test_index,:,:] = Y_test[-1][assessment][:,past_timepoints-1:,:]
                regression_results[assessment][1,test_index,:,:] = predictions[assessment] 
        else: 
            predictions, predictions_std, predicted_remissions, predicted_remissions_std = models[r].predict(X_test[-1], 
                                                                                                     X_static_ts=X_static_ts[-1]) 
        
        for c in criteria:
            R_pred.loc[testing_subs,c] = predicted_remissions[c+'_remission'][:,-1,1]
            R_pred_min.loc[testing_subs,c] = predicted_remissions_std[c+'_remission'][:,-1,1,0]
            R_pred_max.loc[testing_subs,c] = predicted_remissions_std[c+'_remission'][:,-1,1,1]
            
        model_ids.loc[testing_subs] = r
        r += 1
    
    l = []
    for c in criteria:
        l += [c+a for a in ['_Label', '_Prob','_LB','_UB','_Recom']]
    prediction_summary = pd.DataFrame(index=subjects, columns=l + ['Model_ID'] )
    
    for c in criteria:
        prediction_summary.loc[:,c+'_Label'] = remissions[visits[-1]][c+'_remission'].loc[subjects].to_numpy(np.float)
        prediction_summary.loc[:,c+'_Prob'] = R_pred.loc[:,c].to_numpy(np.float)
        prediction_summary.loc[:,c+'_LB'] = R_pred_min.loc[:,c].to_numpy(np.float)
        prediction_summary.loc[:,c+'_UB'] = R_pred_max.loc[:,c].to_numpy(np.float)
    prediction_summary.loc[:,'Model_ID'] = model_ids.iloc[:,0].to_numpy(np.int)
    
    prediction_summary, acc = decision_making(prediction_summary, method=decision_making_method,
                                              criteria=criteria)
    
    ################################## EVALUATION #################################
    
    
    results_original = dict()
    results_modified = dict()
    
    for c in criteria:
        results_original[c] = evaluate_classification(remissions[visits[-1]][c+'_remission'].loc[subjects].to_numpy(np.float), 
                                      prediction_summary[c+'_Prob'].to_numpy(np.float))
        results_modified[c] = evaluate_classification(remissions[visits[-1]][c+'_remission'].loc[subjects].to_numpy(np.float), 
                                      prediction_summary[c+'_FDM'].to_numpy(np.float))
        
    
    results_reg = evaluate_regression(regression_results)
    
    
    ########################### PLOT RESULTS ##################################
    
    
    plot_results(prediction_summary, prognosis, acc, method=decision_making_method, 
                      save_path=save_path_run)
    
    
    ################################ INTERPRETATION ###########################
    
    cfi = dict()
    if with_statics and perform_cfi:
        
        for s,subject in enumerate(subjects): 
          
            cfi[subject] =  counterfactual_interpretation(subject, visits, assessments,
                                                            prediction_summary, 
                                                            features, remissions, 
                                                            models, static_feature_types,
                                                            past=past_timepoints,
                                                            decision=decision_making_method,
                                                            with_dynamics=with_dynamics)
          
            print('%d / %d : Subject %d' %(s+1, len(subjects), subject))
        
    
    ################################ PLOTS ########################################
        
        
        plot_CFI_group_effect(cfi, save_path=save_path_run)
    
    
    ################################ SAVING RESULTS ###############################
    
    prob_true, prob_pred = calibration_curve(prediction_summary['PANNS_Label'].to_numpy(np.float64), 
                                             prediction_summary['PANNS_Prob'].to_numpy(np.float64), n_bins=10)
    cal_disp_original = CalibrationDisplay(prob_true, prob_pred, prediction_summary['PANNS_Prob'].to_numpy(np.float64))
    
    prob_true, prob_pred = calibration_curve(prediction_summary['PANNS_Label'].to_numpy(np.float64), 
                                             prediction_summary['PANNS_FDM'].to_numpy(np.float64), n_bins=10)
    cal_disp_modified = CalibrationDisplay(prob_true, prob_pred, prediction_summary['PANNS_FDM'].to_numpy(np.float64))
    
    
    with open(save_path_run + experiment_name + '.pkl', 'wb') as file:
       pickle.dump({'results_original':results_original, 'results_modified':results_modified,
                    'acc':acc, 'prediction_summary':prediction_summary, 
                    'static_feature_types':static_feature_types, 'visits':visits,
                    'assessments':assessments, 'cfi':cfi, 'results_reg':results_reg,
                    'expert_performance':expert_performance, 'prognosis':prognosis,
                    'regression_results':regression_results, 'criteria':criteria,
                    'cal_disp_original':cal_disp_original, 'cal_disp_modified':cal_disp_modified,
                    }, file, protocol=4)    
   
   
############################### SUMMARIZING RESULTS ###########################


save_path = base_path + 'Results/' + experiment_name + '/'
with open(save_path + '/Run_0/'+ experiment_name + '.pkl', 'rb') as file:
    data = pickle.load(file)

auc = {'original':{key:[] for key in data['criteria']}, 'modified':{key:[] for key in data['criteria']}}
bac = {'original':{key:[] for key in data['criteria']}, 'modified':{key:[] for key in data['criteria']}}
sen = {'original':{key:[] for key in data['criteria']}, 'modified':{key:[] for key in data['criteria']}}
spc = {'original':{key:[] for key in data['criteria']}, 'modified':{key:[] for key in data['criteria']}}
acc = {key:[] for key in data['acc'].keys()}
crt = []

for run in range(repetitions):
    
    save_path_run = save_path + '/Run_' + str(run) + '/'
    with open(save_path_run + experiment_name + '.pkl', 'rb') as file:
        data = pickle.load(file)
    
    for mode in ['original', 'modified']:
        for c in data['criteria']:
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
        
with open(save_path + experiment_name + '.pkl', 'wb') as file:
    pickle.dump({'auc':auc, 'bac':bac, 'sen':sen, 'spc':spc, 'acc':acc, 'crt':crt}, 
                file, protocol=4)   
    
##############################################################################

# import matplotlib.pyplot as plt

# results_dir = Exp.BASE_PATH + 'Results/1/'

# experiments = [name for name in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, name))]

# experiment_names = ['D', 'PPCD', 'PPC', 'PP', 'P','D1', 'PPCD2']
    
# auc_mean = []
# bac_mean = []
# sen_mean = []
# spc_mean = []
# crt_mean = []
# auc_std = []
# bac_std = []
# sen_std = []
# spc_std = []
# crt_std = []

# plt.figure()
# for i,experiment in enumerate(experiments):
#     save_path = results_dir + experiment + '/'
#     with open(save_path + experiment + '.pkl', 'rb') as file:
#         data = pickle.load(file) 
        
#     auc_mean.append(np.mean(data['auc']))
#     bac_mean.append(np.mean(data['bac']))
#     sen_mean.append(np.mean(data['sen']))
#     spc_mean.append(np.mean(data['spc']))
#     crt_mean.append(np.mean(data['crt']))
    
#     auc_std.append(np.std(data['auc']))
#     bac_std.append(np.std(data['bac']))
#     sen_std.append(np.std(data['sen']))
#     spc_std.append(np.std(data['spc']))
#     crt_std.append(np.std(data['crt']))
    

# plt.figure()
# ax = plt.subplot(2,3,1)
# ax.bar(experiment_names, auc_mean, yerr=auc_std, align='center', alpha=0.5, ecolor='black')
# ax.set_title('AUC')
# ax.yaxis.grid(True)
# plt.ylim([0.3,0.8])

# ax = plt.subplot(2,3,2)
# ax.bar(experiment_names, bac_mean, yerr=bac_std, align='center', alpha=0.5, ecolor='black')
# ax.set_title('BAC')
# ax.yaxis.grid(True)
# plt.ylim([0.3,0.8])

# ax = plt.subplot(2,3,3)
# ax.bar(experiment_names, sen_mean, yerr=sen_std, align='center', alpha=0.5, ecolor='black')
# ax.set_title('SEN')
# ax.yaxis.grid(True)
# plt.ylim([0.3,0.9])

# ax = plt.subplot(2,3,4)
# ax.bar(experiment_names, spc_mean, yerr=spc_std, align='center', alpha=0.5, ecolor='black')
# ax.set_title('SPC')
# ax.yaxis.grid(True)
# plt.ylim([0,0.8])

# ax = plt.subplot(2,3,5)
# ax.bar(experiment_names, crt_mean, yerr=crt_std, align='center', alpha=0.5, ecolor='black')
# ax.set_title('CRT')
# ax.yaxis.grid(True)
# plt.ylim([0,0.7])

# plt.tight_layout()


# for experiment in experiments:
#     save_path = Exp.BASE_PATH + 'Results/1/' + experiment + '/'
#     with open(save_path + experiment + '.pkl', 'rb') as file:
#         data = pickle.load(file) 
     
#     plt.figure()§
#     for key in data['acc'].keys():
#         data['acc'][key] = np.nanmean(data['acc'][key])
#     plt.bar(list(data['acc'].keys()), list(data['acc'].values()))

