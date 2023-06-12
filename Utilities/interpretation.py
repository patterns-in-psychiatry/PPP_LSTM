#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:57:19 2022

@author: seykia

"""

from copy import deepcopy
import numpy as np
import pandas as pd
from Models.FLDM import decision_making


def counterfactual_interpretation(subject, visits, assessments, prediction_summary, 
                                  features, remissions, models, static_feature_types,
                                  past= 1, decision='fuzzy', with_dynamics=True,
                                  m=None):
    if with_dynamics:
        future_timepoints = len(visits) - past
    else:
        future_timepoints = 1
        past=1
        assessments=[]
        
    demo = deepcopy(static_feature_types)
    temp = []
    for key in demo.keys():
        if len(demo[key]) == 0:
           temp.append(key)
    for key in temp:
        demo.pop(key)
    if m is None:
        m = prediction_summary.loc[subject]['Model_ID']
    
    a = dict()
    for sft in demo.keys():
        a[sft] = deepcopy(features[visits[-1]][sft].loc[subject].to_numpy())
    
    b = {key: [] for key in demo.keys()}
    c = dict()
    X_check = dict()
    
    for sft in demo.keys():
        b[sft].append(a[sft])
     
    for sft in demo.keys():
        s = 0
        rest_sfts = list(demo.keys())
        rest_sfts.remove(sft)
        for key in demo[sft].keys():
            if demo[sft][key][0] == 'con':
                demo[sft][key].append(len(b[sft]))
                for value in list(np.arange(0,1.05,0.05)):
                    temp = deepcopy(a[sft])
                    temp[s] = value
                    b[sft].append(temp) 
                    for rsft in rest_sfts:
                        b[rsft].append(a[rsft])
                #temp = deepcopy(a)
                #temp[s] = 1
                #b.append(temp)
                demo[sft][key].append(len(b[sft])-1)
                s += 1
            elif demo[sft][key][0] == 'bin':
                demo[sft][key].append(len(b[sft])) 
                temp = deepcopy(a[sft])
                if temp[s] == -1:
                    temp[s] = 1
                else:
                    temp[s] = -1
                b[sft].append(temp)
                for rsft in rest_sfts:
                    b[rsft].append(a[rsft])
                s += 1
            elif demo[sft][key][0] == 'cat':
                demo[sft][key].append(len(b[sft])) 
                e = s + demo[sft][key][1]
                r = list(range(s,e))
                temp = deepcopy(a[sft])
                idx = np.where(temp[r])[0] +  s 
                #r.remove(idx)
                for j in r:
                    temp = deepcopy(a[sft])
                    temp[idx] = 0
                    temp[j] = 1
                    b[sft].append(temp) 
                    for rsft in rest_sfts:
                        b[rsft].append(a[rsft])
                s = e
                demo[sft][key].append(len(b[sft])-1)
                  
    for sft in demo.keys():           
        c[sft] = np.vstack(b[sft])           
    
        c[sft] = c[sft].reshape([c[sft].shape[0],1,c[sft].shape[1]])
    
        X_check[sft] = np.concatenate([c[sft] for i in range(len(visits)-1)], axis=1)
    
    X = dict()
    for key in assessments:
        temp = []
        for p in range(past):
            t = features[visits[p]][key].loc[subject].to_numpy()
            t = np.reshape(t, [1, 1, t.shape[0]])
            temp.append(t)
        temp= np.concatenate(temp, axis=1)
        X[key] = np.repeat(temp, X_check[list(X_check.keys())[0]].shape[0], axis=0)
    
    predictions, predictions_std, predicted_remission, predicted_remission_std= models[m].predict(X, 
                                                                                                 X_static_ts=X_check,
                                                                                                 past= past, future=future_timepoints,
                                                                                                 repetition=50) 
    
    prediction_summary1 = pd.DataFrame(index=range(predicted_remission['PANNS_remission'].shape[0]), 
                                       columns=['Label','PANNS_Prob',
                                                'PANNS_LB','PANNS_UB', 'PANNS_Recom'])
    
    prediction_summary1.loc[:,'PANNS_Prob'] = predicted_remission['PANNS_remission'][:,-1,1]
    
    prediction_summary1.loc[:,'PANNS_LB'] = predicted_remission_std['PANNS_remission'][:,-1,1,0]
    prediction_summary1.loc[:,'PANNS_UB'] = predicted_remission_std['PANNS_remission'][:,-1,1,1]
    prediction_summary1.loc[:,'PANNS_Label'] = remissions[visits[-1]]['PANNS_remission'].loc[subject].to_numpy()
    
    prediction_summary1, acc = decision_making(prediction_summary1, method=decision, plot=False)
    
    if decision=='fuzzy':
        target_prob = 'PANNS_FDM'
    elif decision=='plain':
        target_prob = 'PANNS_Prob'
    
    feature_effect = dict.fromkeys(demo)
    for sft in demo.keys():
        feature_effect[sft] = dict.fromkeys(demo[sft])
        for key in demo[sft].keys():
            feature_effect[sft][key] = dict()
            if demo[sft][key][0] == 'con':
                feature_effect[sft][key]['predictions'] = prediction_summary1[target_prob][demo[sft][key][2]:demo[sft][key][3]+1].to_numpy()
                effect_size = np.abs(prediction_summary1[target_prob][demo[sft][key][3]] - \
                        prediction_summary1[target_prob][demo[sft][key][2]])
                #effect_size = np.max(prediction_summary1[target_prob].iloc[demo[sft][key][2]:demo[sft][key][3]+1]) - \
                #        np.min(prediction_summary1[target_prob].iloc[demo[sft][key][2]:demo[sft][key][3]+1])
            elif demo[sft][key][0] == 'cat':
                feature_effect[sft][key]['predictions'] = prediction_summary1[target_prob][demo[sft][key][2]:demo[sft][key][3]+1].to_numpy() 
                temp = feature_effect[sft][key]['predictions'] - \
                                                prediction_summary1[target_prob].iloc[0]
                effect_size = np.max(np.abs(temp))
                feature_effect[sft][key]['index'] = np.argmax(np.abs(temp))
                #effect_size = np.max(prediction_summary1[target_prob].iloc[demo[sft][key][2]:demo[sft][key][3]+1]) - \
                #        np.min(prediction_summary1[target_prob].iloc[demo[sft][key][2]:demo[sft][key][3]+1])
            elif demo[sft][key][0] == 'bin':
                feature_effect[sft][key]['predictions'] = np.array([prediction_summary1[target_prob].iloc[0], 
                                                          prediction_summary1[target_prob][demo[sft][key][2]]])
                effect_size = np.abs(prediction_summary1[target_prob].iloc[demo[sft][key][2]] - \
                        prediction_summary1[target_prob].iloc[0])
            feature_effect[sft][key]['effect'] =  feature_effect[sft][key]['predictions'] - prediction_summary1[target_prob].iloc[0]
            feature_effect[sft][key]['effect_size'] = effect_size
            
    return feature_effect


def compile_patient_recom(static_features, cfi, threshold=0.05):

    
    patients = cfi.keys()
    fields = {'demographics':{'V1_MARRIED':['bin'],'V1_OCCUPATION':['bin'],
                                'V1_LIVING_ALONE':['bin'],'V1_DWELLING':['cat'],
                                'V1_INCOME_SRC':['cat'],'V1_LIVING_INVIRONMT':['cat']},
                     'lifestyle':{'V2_RECDRUGS':['bin'], 'V2_CAFFEINE_CUPS':['con']}, 
                     'somatic':{'V1_PE_WEIGHT_KG':['con'],'V1_PE_WAIST_CM':['con'],
                            'V1_PE_SBP':['con'],'V1_PE_DBP':['con'],'BMI':['con']}}
    names = {'V1_MARRIED':'marriage status', 'V1_OCCUPATION':'occupation status',
             'V1_LIVING_ALONE':'living alone condition','V1_DWELLING':'Type of dwelling',
             'V1_INCOME_SRC':'main source of income','V1_LIVING_INVIRONMT':'living environment',
             'V2_RECDRUGS':'recreational drugs usage status','V2_CAFFEINE_CUPS':'coffeine usage status',
             'V1_PE_WEIGHT_KG':'weight','V1_PE_WAIST_CM':'waist','V1_PE_SBP':'systolic blood pressure',
             'V1_PE_DBP':'diastolic blood pressure','BMI':'BMI'}
    values = {'V1_MARRIED':["'NO'","'YES'"], 'V1_OCCUPATION':["'NO'","'YES'"],
             'V1_LIVING_ALONE':["'NO'","'YES'"],'V1_DWELLING':["'HOUSE'","'APARTMENT'","'ROOM'",
                                                               "'HOUSE with supervision'","'DORMITORY'",
                                                               "'TEMPORARY'","'OTHER'"],
             'V1_INCOME_SRC':["'EMPLOYMENT'","'PARENTS'","'SOCIAL SECURITY'","'UNEMPLOYMENT BENEFIT'","'OTHER'"],
             'V1_LIVING_INVIRONMT':["'BIG CITY'","'MEDIUM CITY'","'SMALL CITY'","'VILLAGE'"],
             'V2_RECDRUGS':'recreational drugs usage status','V2_CAFFEINE_CUPS':'coffeine usage status',
             'V1_PE_WEIGHT_KG':[],'V1_PE_WAIST_CM':[],'V1_PE_SBP':[],'V1_PE_DBP':[],'BMI':[]}
    
    text=dict()
    for patient in patients:
        text[patient]=''
        for field in fields:
            for recom in fields[field]:
                if cfi[patient][field][recom]['effect_size']>=threshold:
                    if fields[field][recom][0] == 'bin':
                        if np.sum(cfi[patient][field][recom]['effect']) > 0:
                            text[patient] += 'Changing ' + names[recom] + ' improves the chance of remission by ' + str(cfi[patient][field][recom]['effect_size']) + '. '
                        else:
                            text[patient] += 'Maintaining ' + names[recom] + ' has improved the chance of remission by ' + str(cfi[patient][field][recom]['effect_size']) + '. '   
                    elif fields[field][recom][0] == 'con':
                        if cfi[patient][field][recom]['effect'][0] > 0 or cfi[patient][field][recom]['effect'][-1] < 0:
                            text[patient] += 'Reduce in ' + names[recom] + ' can improve the chance of remission by ' + str(cfi[patient][field][recom]['effect_size']) + '. '   
                        elif cfi[patient][field][recom]['effect'][0] < 0 or cfi[patient][field][recom]['effect'][-1] > 0:
                            text[patient] += 'Increase in ' + names[recom] + ' can improve the chance of remission by ' + str(cfi[patient][field][recom]['effect_size']) + '. '
                    elif fields[field][recom][0] == 'cat':
                        if (cfi[patient][field][recom]['effect'][cfi[patient][field][recom]['index']]>0 and 
                            values[recom][int(static_features[field].loc[patient][recom]-1)]!=values[recom][cfi[patient][field][recom]['index']]):
                            text[patient] += 'Changing ' + names[recom] + ' from ' +  values[recom][int(static_features[field].loc[patient][recom]-1)] + \
                                ' to option ' + values[recom][cfi[patient][field][recom]['index']] +' improves the chance of remission by ' + str(cfi[patient][field][recom]['effect_size']) + '. '
                        else:
                            text[patient] += 'Maintaining ' + names[recom] + ' condition ' + values[recom][int(static_features[field].loc[patient][recom]-1)] + ' has improved the chance of remission by ' + str(cfi[patient][field][recom]['effect_size']) + '. '
        if text[patient]=='':
            text[patient] += 'The model has no recommendation for this patient.'
                
    return text                          