#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:40:49 2020

@author: seykia
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import scipy.io
from sklearn.impute import SimpleImputer, KNNImputer
from Utilities.scaler import scaler
import json
import copy


def merge_all_data(data_path):
    dfs = []
    temp = pd.read_csv(data_path + 'inclusion.csv', sep=';')
    temp = temp.rename(columns={'code': 'Code'})
    temp = temp.set_index('Code')
    dfs.append(temp)
    visits = ['1', '2', '3', '4', '5', '6', '7', '8', '10', '12', '16', '20']

    for visit in visits:
        if visit == '1':
            data = pd.read_csv(data_path + 'Visit_' + visit + '.csv', sep=',')
        else:
            data = pd.read_csv(data_path + 'Visit_' + visit + '.csv', sep=';')
        data = data.set_index('Code')
        dfs.append(data)

    merged = dfs[0]

    for i in range(1, len(dfs)):
        merged = pd.merge(merged, dfs[i], how='left', on='Code')

    merged.to_csv(data_path + 'merged_data.csv')

    return 1


def read_cytokines(data_path:str, features:list) -> pd.DataFrame:
    """

    :return: The preprocessed DataFrame
    """

    df_cytokines = pd.read_pickle(data_path + 'cytokines_umcu.pkl')

    ## If the UMCU dataset is used, do the following preprocessing steps
    # Covert '< LLOD' strings to numeric values
    df_cytokines = preprocess_LLOD(df_cytokines, method='min_values')

    # Convert datatype of every feature to numeric or else nan
    df_cytokines = df_cytokines.apply(pd.to_numeric, errors='coerce')

    # Drop subjects that do not have any measurement (only nan's')
    df_cytokines = df_cytokines.dropna(axis=0, how='all')

    # Drop features that do not have any measurement (only nan's')
    df_cytokines = df_cytokines.dropna(axis=1, how='all')

    ## Dealing with Nan's
    # Minimum amount of non-NA values that is acceptable
    # min_amount_non_NA = int((1 - Exp.PROP_MISSING) * (df_cytokines.shape[0] + 1))

    # Drop columns that don't reach the non-NA treshold
    # df_cytokines = df_cytokines.dropna(axis=1, thresh=min_amount_non_NA)

    return df_cytokines[features]



def read_MATLAB_data(file_path:str, df_UMCU:pd.DataFrame) -> pd.DataFrame:
    """
    Temporary utility function (can be deleted later on)
    :param file_path: Absolute file path of .mat file
    :param df_UMCU:
    :return:
    """

    mat = scipy.io.loadmat(file_path)

    # Actual data/values
    X = mat['X_cytokines']
    # X = np.concatenate((mat['X_cytokines'], mat['X_panss']), axis=1)

    # Column names
    columns = []
    columns_cytokines = mat['cytokines'].flatten()
    # columns_panss = mat['panss'].flatten()

    # Convert ndarray of ndarrays to list of strings
    for column in columns_cytokines:
        columns.append(np.array2string(column)[2:-2])

    # for column in columns_panss:
        # columns.append(np.array2string(column)[2:-2])

    # Index
    subjects = list(mat['patients'].flatten())

    # Rebuild French DataFrame for comparison with ours
    df_French = pd.DataFrame(data=X, columns=columns, index=subjects)
    assert not df_French.isnull().values.any(), 'data contain missing values'

    return df_French


def preprocess_LLOD(df_cytokines: pd.DataFrame, method:str) -> pd.DataFrame:
    """
    Converts the string values of "< LLOD" (i.e. smaller than lower limit of detection) to a numeric value
    TODO: Finish the function. Mailed the French if they have the LLOD values, as the MATLAB code they sent did not contain it
    :param df_cytokines:
    :param method:
    :return:
    """

    if method == 'min_values':
        for feature in df_cytokines.columns:
            # Get the min value of the fßeature
            df_biological_copy = df_cytokines.copy(deep=True)
            df_biological_copy = df_biological_copy.apply(pd.to_numeric, errors='coerce')
            feature_series = df_biological_copy[feature].dropna()
            min_value = min(feature_series)

            # Feature's LLOD replacement value
            LLOD = min_value / 2

            # Replace the '< LLOD' with the newly computed LLOD value
            df_cytokines[feature] = df_cytokines[feature].replace(to_replace='.*< LLOD.*', value=LLOD, regex=True)

    elif method == 'french':
        NotImplemented

    # Check for remaining LLOD values
    for column in df_cytokines.columns:
        if df_cytokines[column].dtype == 'object' and df_cytokines[column].str.contains('LLOD', regex=True).any():
            print(f'Column with LLOD strings: {column}')

    # Check if there are any strings left
    assert (df_cytokines.dtypes != 'object').all(), 'There are still some strings left in the DataFrame'

    return df_cytokines


def read_sites(data_path: str) -> pd.DataFrame:
    """
    Returns a DataFrame with the site data for patients
    :param data_path: Absolute path to the data files
    :return: DataFrame with the features of interest
    """

    data = pd.read_csv(data_path + 'merged_data.csv', sep=',')
    data = data.set_index('Code')
    sites = pd.DataFrame(data[['OrganisationName']])
    sites['site_id'] = 0
    sites.loc[sites["OrganisationName"] == "15_Tangent Data_Unit 9", "OrganisationName"] = "15_Tangent Data_Unit"
    sites.loc[sites["OrganisationName"] == "15_Tangent Data_Unit 1", "OrganisationName"] = "15_Tangent Data_Unit"
    sites.loc[sites["OrganisationName"] == "15_Tangent Data_Unit 3", "OrganisationName"] = "15_Tangent Data_Unit"
    sites.loc[sites["OrganisationName"] == "15_Tangent Data_Unit 12", "OrganisationName"] = "15_Tangent Data_Unit"
    sites.loc[sites["OrganisationName"] == "15_Tangent Data_Unit 13", "OrganisationName"] = "15_Tangent Data_Unit"
    sites.loc[sites["OrganisationName"] == "15_Tangent Data_Unit 4", "OrganisationName"] = "15_Tangent Data_Unit"
    sites.loc[sites["OrganisationName"] == "15_Tangent Data_Arad", "OrganisationName"] = "15_Tangent Data_Unit"
    u = list(sites['OrganisationName'].unique())
    for i in range(sites.shape[0]):
        sites.iloc[i,1] = u.index(sites.iloc[i,0])
    
    return sites


def read_demographics(data_path: str, demographics: dict, include_missing=False) -> pd.DataFrame:
    """
    Returns a DataFrame with the features included in `demographics`
    :param data_path: Absolute path to the data files
    :param demographics: Dict of the features of interest and their associated datatype
    :return: DataFrame with the features of interest
    """

    sw = 0
    d = list(demographics.keys())
    demo_info = pd.DataFrame()

    data = pd.read_csv(data_path + 'merged_data.csv', sep=',')
    data = data.set_index('Code')
    data = data.replace(' ', np.NaN)

    if 'BMI' in d:
        demo_info = pd.DataFrame(data[['V1_PE_WEIGHT_KG', 'V1_PE_HEIGHT_CM']])
        for c in demo_info.columns:
            demo_info[c] = pd.to_numeric(demo_info[c], errors='coerce')
        demo_info['BMI'] = demo_info['V1_PE_WEIGHT_KG'] / (demo_info['V1_PE_HEIGHT_CM'] / 100) ** 2
        demo_info['BMI'].iloc[np.where(demo_info['BMI'] > 50)] = np.nan
        demo_info = demo_info.drop(columns=['V1_PE_WEIGHT_KG', 'V1_PE_HEIGHT_CM'])
        d.remove('BMI')
        sw = 1

    if 'avg_dosage' in d:
        data1 = pd.read_csv(data_path + 'WP2_Study_medication_phase1_21jun2019.csv')
        data1 = data1.set_index('Code')
        average_dosage = pd.DataFrame(index=list(pd.unique(data1.index)), columns=['avg_dosage'])
        for code in list(pd.unique(data1.index)):
            a = data1.loc[code]
            average_dosage['avg_dosage'].loc[code] = (a['SMED_AM_DOSE'] * a['days']).sum() / a['days'].sum()
        if sw == 1:
            demo_info = demo_info.join(pd.DataFrame(average_dosage[['avg_dosage']]))
        else:
            demo_info = pd.DataFrame(average_dosage[['avg_dosage']])
        for c in demo_info.columns:
            demo_info[c] = pd.to_numeric(demo_info[c], errors='coerce')
        d.remove('avg_dosage')
        sw = 1

    if sw == 1:
        if include_missing:  # Difference with v 0.0
            demo_info = demo_info.join(data[d], how='outer')
        else:
            demo_info = demo_info.join(data[d], how='inner')
    else:
        demo_info = data[d]

    for c in demo_info.columns:
        demo_info[c] = pd.to_numeric(demo_info[c], errors='coerce')
    demo_info = demo_info.sort_index()
    demo_info = demo_info.astype('float32')

    return demo_info


def read_static_features(data_path, static_feature_types: dict, include_missing=False) -> dict:
    """
    This function takes as input the relevant features types (e.g. demographics, lifestyle, cytokines) and
    returns a dict of features and associated DataFrames.
    The features included in the DataFrame depend on the features listed
    in static_features_types.

    :param Dict of features type(s), where each feature type holds the variable name and datatype
    :return: Dict of features and their associated DataFrame
    """

    static_feature = dict()
    feature_types = list(static_feature_types.keys())

    for feature_type in feature_types:
        if feature_type == 'cytokines':
            features = list(static_feature_types['cytokines'].keys())
            static_feature['cytokines'] = read_cytokines(data_path, features=features)
        else:
            static_feature[feature_type] = read_demographics(data_path, 
                                                             static_feature_types[feature_type],
                                                             include_missing)

    return static_feature


def read_prognosis(data_path, subjects=None) -> pd.DataFrame:
    """
    Converts the scale of the prognosis acquired at V1 from 1-7 to [0, 1] and adds a *recommendation* feature to
    the prognosis DataFrame with the following categories:

    DR: definite remission, PR: probably remission, UR: uncertain remission, US: uncertain ?, UN: uncertain
    no remission, PN: probable no-remission, DN: definite ro-remission

    :param subjects: List of subjects' codes/id's
    :return: DataFrame with for each subject the rescaled original V1 prognosis and a new recommendation
    category
    """

    data = pd.read_csv(data_path + 'Visit_1.csv', sep=',')
    data = data.set_index('Code')
    prognosis = data['V1_PROGNOSIS']
    prognosis = pd.to_numeric(prognosis, errors='coerce')
    prognosis = prognosis.sort_index()
    prognosis = pd.DataFrame(prognosis)
    prognosis['Recommendation'] = np.nan
    recoms = ['DR', 'PR', 'UR', 'US', 'UN', 'PN', 'DN']
    for i in range(1, 8):
        prognosis['Recommendation'].iloc[np.where(prognosis['V1_PROGNOSIS'] == i)] = recoms[i - 1]
    prognosis['V1_PROGNOSIS'] = (1 - prognosis[
        'V1_PROGNOSIS'] * 1.0 / 7.0) + 1.0 / 14.0  
    if subjects is not None:
        prognosis = prognosis.loc[subjects]
    return prognosis


def read_dynamic_features(data_path, visits, assessments=['PANNS'], include_missing=False) -> pd.DataFrame:
    """
    Get DataFrames from participants' subsequent visits after their initial baseline visit. Current dynamic measures
    are the measures `PANSS`, `PSP` and `CGI`.

    :return: DataFrame for each visit (e.g. V2) and for each of the measurements (e.g. PANSS)
    """

    measures = dict.fromkeys(visits)
    for key in measures.keys():
        measures[key] = dict.fromkeys(assessments, pd.DataFrame())

    for visit in visits:
        data = pd.read_csv(data_path + 'Visit_' + visit + '.csv', sep=';')
        data = data.set_index('Code')
        S = pd.Series(data.columns)
        for assessment in assessments:
            measures[visit][assessment] = data[data.columns[S.str.contains(assessment + '_')]]
            for c in measures[visit][assessment].columns:
                measures[visit][assessment][c] = pd.to_numeric(measures[visit][assessment][c], errors='coerce')
            if not include_missing:
                measures[visit][assessment] = measures[visit][assessment].dropna()
            if visit=='2' and assessment=='CGI':
                measures[visit][assessment]['V2_CGI_IMPROV'] = 0

    return measures


def feature_extraction(dynamic_features: dict, base_visit='2',
                       static_features=None, static_feature_types=None,
                       aggregation='union', include_missing=False):
    """
    TODO: Finish documentation after having used this functions more

    :param dynamic_features: Dict of dynamic_features for each visit
    :param base_visit:
    :param static_features:
    :param static_feature_types:
    :param aggregation: Whether to only include subjects that have a measurement for every feature (i.e. intersection)
     or subjects that have at least one measurement/value for one of the features (i.e. union)
    :return:
    """
    dynamic_features1 = copy.deepcopy(dynamic_features)
    visits = list(dynamic_features1.keys())
    assessments = list(dynamic_features1[visits[0]].keys())

    features = dict.fromkeys(visits)

    for key in features.keys():
        subjects = []
        for assessment in assessments:
            subjects.append(list(dynamic_features1[key][assessment].index))
        if aggregation == 'intersection':
            s = list(set(subjects[0]).intersection(*subjects))
        elif aggregation == 'union':
            s = list(set(subjects[0]).union(*subjects))
        s.sort()
        for assessment in assessments:
            dynamic_features1[key][assessment] = dynamic_features1[key][assessment].reindex(s)

    if 'PSP' in assessments:
        for key in features.keys():
            try:
                dynamic_features1[key]['PSP']['V' + str(key) + '_PSP_SCORE'] = dynamic_features1[key]['PSP'][
                                                                          'V' + str(key) + '_PSP_SCORE'] / 10
            except:
                pass
    
    
    if not include_missing:
        imputer = dict.fromkeys(assessments)
        for assessment in assessments:
            imputer[assessment] = SimpleImputer(strategy='median')
            a = imputer[assessment].fit_transform(dynamic_features1[base_visit][assessment].values)

    sclr = dict()
    for counter, key in enumerate(features.keys()):
        features[key] = dict()
        for assessment in assessments:
            if include_missing:
                features[key][assessment] = dynamic_features1[key][assessment]
            else:
                a = imputer[assessment].transform(dynamic_features1[key][assessment].values)
                features[key][assessment] = pd.DataFrame(a, index=dynamic_features1[key][assessment].index)

        if static_features is not None:

            for k in static_features.keys():
                d_org = static_features[k]
                d = d_org.reindex(features[key][assessments[0]].index)
                
                temp = []
                for c, col in enumerate(d.columns):
                    if static_feature_types[k][col][0] == 'bin':
                        if not include_missing:
                            imp = SimpleImputer(strategy='most_frequent')
                            d[col] = imp.fit_transform(d[col].values.reshape(-1, 1))
                        d[col].iloc[np.where(d[col] == 2)] = -1 #0
                        temp.append(d[col].values.reshape(-1, 1))
                        static_feature_types[k][col] = ['bin', 2]
                    elif static_feature_types[k][col][0] == 'cat':
                        if not include_missing:
                            imp = SimpleImputer(strategy='most_frequent')
                            d[col] = imp.fit_transform(d[col].values.reshape(-1, 1))
                        ohe = OneHotEncoder(sparse=False, categories=[
                            list(range(1, int(d_org[col].max() + 1)))], handle_unknown='ignore')  # , categories=list(range(int(d[col].max()+1)))
                        temp.append(ohe.fit_transform(d[col].values.reshape(-1, 1)))
                        static_feature_types[k][col] = ['cat', temp[c].shape[1]]
                    elif static_feature_types[k][col][0] == 'con':
                        if not include_missing:
                            imp = SimpleImputer(strategy='median')
                            d[col] = imp.fit_transform(d[col].values.reshape(-1, 1))
                        if counter == 0:
                            sclr[col] = scaler(scaler_type='robminmax')
                            sclr[col].fit(d[col].values.reshape(-1, 1))
                        temp.append(sclr[col].transform(d[col].values.reshape(-1, 1)))
                        static_feature_types[k][col] = ['con', 1]
                features[key][k] = pd.DataFrame(np.concatenate(temp, axis=1), index=d.index)

    return features, static_feature_types


def compute_andreasen_remission_labels(measures):
    """

    :param measures:
    :return:
    """
    visits = list(measures.keys())
    
    panss_remission_criteria = ['PANNS_P1', 'PANNS_P2', 'PANNS_P3', 'PANNS_N1', 'PANNS_N4',
                                'PANNS_N6', 'PANNS_G5', 'PANNS_G9']
    remission = dict.fromkeys(visits)
    for v in visits:
        remission[v] = dict()
        remission[v]['PANNS_remission'] = pd.DataFrame(data=np.zeros([measures[v]['PANNS'].shape[0], ]),
                                                       index=measures[v]['PANNS'].index)
        for i in range(measures[v]['PANNS'].shape[0]):
            sw = 1
            for cr in panss_remission_criteria:
                if (measures[v]['PANNS']['V' + v + '_' + cr].iloc[i] > 3 or 
                    np.isnan(measures[v]['PANNS']['V' + v + '_' + cr].iloc[i])):
                    sw = 0
                    break;
            remission[v]['PANNS_remission'].iloc[i] = sw
        subjects = measures[v][list(measures[v].keys())[0]].index
        remission[v]['PANNS_remission'] = remission[v]['PANNS_remission'].loc[subjects]

    return remission


def compute_output_labels(measures, labels=['PANNS']):
    """

    :param measures:
    :return:
    """
    visits = list(measures.keys())
    
    if 'PANNS' in labels:
        outputs = compute_andreasen_remission_labels(measures)
        
    if 'PSP' in labels:
        for v in visits:
            outputs[v]['PSP_remission'] = pd.DataFrame(data=np.zeros([measures[v]['PSP'].shape[0], ]),
                                                           index=measures[v]['PSP'].index)
            for i in range(measures[v]['PSP'].shape[0]):
                if measures[v]['PSP']['V' + v + '_PSP_SCORE'].iloc[i] > 70:
                    outputs[v]['PSP_remission'].iloc[i] = 1
            subjects = measures[v][list(measures[v].keys())[0]].index
            outputs[v]['PSP_remission'] = outputs[v]['PSP_remission'].reindex(subjects)
            outputs[v]['PSP_remission'][np.isnan(outputs[v]['PSP_remission'])] = 0
            
    if 'CGI' in labels:
        for v in visits:
            outputs[v]['CGI_remission'] = pd.DataFrame(data=np.zeros([measures[v]['CGI'].shape[0], ]),
                                                           index=measures[v]['CGI'].index)
            for i in range(measures[v]['CGI'].shape[0]):
                if measures[v]['CGI']['V' + v + '_CGI_SEVERITY'].iloc[i] < 5:
                    outputs[v]['CGI_remission'].iloc[i] = 1
            subjects = measures[v][list(measures[v].keys())[0]].index
            outputs[v]['CGI_remission'] = outputs[v]['CGI_remission'].reindex(subjects)
            outputs[v]['CGI_remission'][np.isnan(outputs[v]['CGI_remission'])] = 0
    return outputs


def panss_to_labels(panss):
    """

    :param panss:
    :return:
    """
    panss = np.round(panss)
    labels = np.zeros([panss.shape[0], panss.shape[1], 1])
    idx = [0, 1, 2, 8, 11, 13, 19, 23]
    for i in range(panss.shape[0]):
        for j in range(panss.shape[1]):
            sw = 1
            for cr in idx:
                if panss[i, j, cr] > 3:
                    sw = 0
                    break;
            labels[i, j, 0] = sw

    return labels


def compute_remission_score(measures, instrument='PANNS', remission_criteria=None,
                            pca=False):
    """

    :param measures:
    :param instrument:
    :param remission_criteria:
    :param pca:
    :return:
    """

    visits = list(measures.keys())

    if remission_criteria is None:
        instrument = 'PANNS'
        remission_criteria = ['PANNS_P1', 'PANNS_P2', 'PANNS_P3', 'PANNS_N1', 'PANNS_N4',
                              'PANNS_N6', 'PANNS_G5', 'PANNS_G9']
    remission = dict.fromkeys(visits)
    for v in visits:
        cr = ['V' + v + '_' + c for c in remission_criteria]

        if v == '2' and pca:
            pca = PCA(n_components=1)
            remission[v] = pd.DataFrame(data=pca.fit_transform(measures[v][instrument][cr]),
                                        index=measures[v][instrument].index)
        elif pca:
            remission[v] = pd.DataFrame(data=pca.fit_transform(measures[v][instrument][cr]),
                                        index=measures[v][instrument].index)
        else:
            remission[v] = pd.DataFrame(data=measures[v][instrument][cr],
                                        index=measures[v][instrument].index)
    return remission


def compute_classifier_inputs(features, remissions, assessments=['PANNS'],
                              criterion='PANNS', static_feature_types=None,
                              feature_visits=['2'], label_visit='5', 
                              missing_values='impute'):
    """
    
    :param features: DESCRIPTION
    :type features: TYPE
    :param remissions: DESCRIPTION
    :type remissions: TYPE
    :param assessments: DESCRIPTION, defaults to ['PANNS']
    :type assessments: TYPE, optional
    :param criterion: DESCRIPTION, defaults to 'PANNS'
    :type criterion: TYPE, optional
    :param static_feature_types: DESCRIPTION, defaults to None
    :type static_feature_types: TYPE, optional
    :param feature_visits: DESCRIPTION, defaults to ['2']
    :type feature_visits: TYPE, optional
    :param label_visit: DESCRIPTION, defaults to '5'
    :type label_visit: TYPE, optional
    :param missing_values: DESCRIPTION, defaults to 'impute'
    :type missing_values: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """

    Y = remissions[label_visit][criterion+'_remission']
    
    temp_features = dict()
    for feature in assessments:
        temp_features[feature] = list()
        for visit in feature_visits:
            temp_features[feature].append(features[visit][feature].reindex(features[label_visit][feature].index))
        temp_features[feature] = pd.concat(temp_features[feature], axis=1)
        
    if static_feature_types is not None:
        for feature in static_feature_types.keys():
            temp_features[feature] = features[feature_visits[0]][feature].reindex(features[label_visit][feature].index)

    X = pd.concat(temp_features, axis=1)
    subjects = X.index
    
    if missing_values == 'remove':
        X = X.dropna()
        Y = Y.loc[X.index]
        X = X.values
    elif missing_values == 'impute':
        X = X.values
        imp =  SimpleImputer(missing_values=np.nan, strategy='median')
        X = imp.fit_transform(X)
    elif missing_values == 'keep':
        X = X.values
    
    Y = Y.values

    return X, Y, subjects


def compute_LSTM_inputs(features, remission, feature_types=['ALL'],
                        feature_visits=['2'], label_visits=['5'],
                        last_visit='5', demo_info=None):
    """

    :param features:
    :param remission:
    :param feature_types:
    :param feature_visits:
    :param label_visits:
    :param last_visit:
    :param demo_info:
    :return:
    """

    if feature_types[0] == 'ALL':
        feature_types = ['PANNS', 'PSP', 'CDSS', 'SWN']

    X = list()
    for v, visit in enumerate(feature_visits):
        a = list()
        for f in feature_types:
            a.append(features[visit][f].loc[features[last_visit][f].index])
        if demo_info is not None:
            a.append(demo_info.loc[features[last_visit][f].index])

        X.append(pd.concat(a, axis=1).dropna())

    index_intersect = pd.concat(X, axis=1).dropna().index
    for i in range(len(X)):
        X[i] = X[i].loc[index_intersect].values

    Y = list()
    for visit in label_visits:
        ohe = OneHotEncoder(sparse=False)
        temp = ohe.fit_transform(remission[visit].loc[index_intersect].values)
        Y.append(temp)

    # Y = Y.loc[index_intersect].values
    # Y = ohe.fit_transform(Y)
    X = np.stack(X, axis=2)
    X = np.transpose(X, [0, 2, 1])

    Y = np.stack(Y, axis=2)
    Y = np.transpose(Y, [0, 2, 1])

    return X, Y


def data_simulation(n_samples, measures, criteria, static_features=None, 
                    static_feature_types=None):
    """

    :param n_samples:
    :param measures:
    :param static_features:
    :param static_feature_types:
    :return:
    """
    
    visits = list(measures.keys())
    assessments = list(measures[visits[0]].keys())
    simulated_measures = dict()
    for v, visit in enumerate(visits):
        simulated_measures[visit] = dict()
        if v == 0:
            for a in assessments:
                mx = np.nanmax(measures[visit][a].to_numpy()) + 1
                mn = np.nanmin(measures[visit][a].to_numpy())
                simulated_measures[visit][a] = pd.DataFrame(np.random.randint(mn, mx,
                                                                              [n_samples, measures[visit][a].shape[1]]),
                                                            columns=measures[visit][a].columns)
        else:
            for a in assessments:
                mx = np.nanmax(measures[visits[0]][a].to_numpy())
                mn = np.nanmin(measures[visits[0]][a].to_numpy())
                temp = simulated_measures[visits[v - 1]][a].to_numpy() + \
                       np.random.randint(-2, 2, [n_samples, measures[visit][a].shape[1]])
                temp[temp < mn] = mn
                temp[temp > mx] = mx
                simulated_measures[visit][a] = pd.DataFrame(temp, columns=measures[visit][a].columns)

    if static_features is not None:
        simulated_static = dict()
        for sf in static_features.keys():
            simulated_static[sf] = pd.DataFrame(columns=static_features[sf].columns)

            for c in static_features[sf].columns:
                mn = np.round(np.nanmin(static_features[sf][c]))
                mx = np.round(np.nanmax(static_features[sf][c])) + 1
                simulated_static[sf][c] = np.random.randint(mn, mx, [n_samples, ])

        simulated_features, _ = feature_extraction(simulated_measures,
                                                   static_features=simulated_static,
                                                   static_feature_types=static_feature_types)
    else:
        simulated_features, _ = feature_extraction(simulated_measures,
                                                   static_features=static_features,
                                                   static_feature_types=static_feature_types)

    simulated_remissions = dict()
    simulated_remissions = compute_output_labels(simulated_measures, labels=criteria)

    return simulated_features, simulated_remissions


def create_static_features_json(data_path, study):
    """

    :param study:
    :return:
    """

    if study == 'test':
        static_feature_types = {'demographics': {'sexe': ['bin'], 'V1_DOV_AGE': ['con'],
                                                 'V1_RACE': ['cat'], 'V1_CNTR_RES': ['bin'],
                                                 'V1_MARRIED': ['bin'], 'V1_DIVORCED': ['bin'],
                                                 'V1_OCCUPATION': ['bin'], 'V1_OCC_TYPE': ['cat'],
                                                 'V1_OCC_PREV': ['bin'], 'V1_OCC_PREV_TYPE': ['cat'],
                                                 'V1_OCC_FATH': ['cat'], 'V1_OCC_MOTH': ['cat'],
                                                 'V1_LIVING_ALONE': ['bin'], 'V1_DWELLING': ['cat'],
                                                 'V1_INCOME_SRC': ['cat'], 'V1_LIVING_INVIRONMT': ['cat']},
                                'diagnosis': {'V1_INC_DSM': ['cat'], 'V1_INC_PSYCH_DUR': ['con'],
                                              'V1_TRSETTING': ['cat'], 'V1_PSYCHINTV': ['bin'],
                                              'V1_PROGNOSIS': ['cat'], 'V2_HOSP': ['bin'],
                                              'V2_SAE': ['bin']},
                                'lifestyle': {'V1_RECDRUGS': ['bin'], 'V2_RECDRUGS': ['bin'],
                                              'V2_CAFFEINE_CUPS': ['con'], 'V2_LST_CAFFEINE': ['cat'],
                                              'V2_ALC_12M': ['cat'], 'V2_ALC_NEVER': ['bin'],
                                              'V2_SMOKE': ['cat']},
                                'somatic': {'V1_ECG_NORM': ['bin'], 'V1_PE_HEIGHT_CM': ['con'],
                                            'V1_PE_WEIGHT_KG': ['con'], 'V1_PE_WAIST_CM': ['con'],
                                            'V1_PE_HIP_CM': ['con'], 'V1_PE_SBP': ['con'],
                                            'V1_PE_DBP': ['con'], 'V1_PE_PULSE': ['con'],
                                            'V2_LSTMEAL': ['cat'], 'V2_LSTMEAL_TYPE': ['cat'],
                                            'BMI': ['con']},
                                'treatments': {'avg_dosage': ['con']},
                                'mini': {'V1_MINI_A_DEPR_CURR': ['bin'], 'V1_MINI_A_DEPR_RECURR': ['bin'],
                                         'V1_MINI_A_DPMOOD_GNR_CURR': ['bin'],
                                         'V1_MINI_A_DPMOOD_GNR_PAST': ['bin'], 'V1_MINI_A_DPMOOD_SUBST_CURR': ['bin'],
                                         'V1_MINI_A_DPMOOD_SUBST_PAST': ['bin'],
                                         'V1_MINI_A_MELANCH_CURR': ['bin'], 'V1_MINI_A_MELANCH_RECURR': ['bin'],
                                         'V1_MINI_B_DYST_CURR': ['bin'],
                                         'V1_MINI_B_DYST_PAST': ['bin'], 'V1_MINI_C_CURR': ['bin'],
                                         'V1_MINI_C_RISK': ['cat'],
                                         'V1_MINI_D_MANIC_CURR': ['bin'], 'V1_MINI_D_MANIC_PAST': ['bin'],
                                         'V1_MINI_D_HYPOM_CURR': ['bin'],
                                         'V1_MINI_D_HYPOM_PAST': ['bin'], 'V1_MINI_D_BIPOLII_CURR': ['bin'],
                                         'V1_MINI_D_BIPOLII_PAST': ['bin'],
                                         'V1_MINI_D_MANIC_GNR_CURR': ['bin'], 'V1_MINI_D_MANIC_GNR_PAST': ['bin'],
                                         'V1_MINI_D_MANIC_SUBST_CURR': ['bin'],
                                         'V1_MINI_D_MANIC_SUBST_PAST': ['bin'], 'V1_MINI_E_PANIC_CURR': ['bin'],
                                         'V1_MINI_E_PANIC_LIFE': ['bin'],
                                         'V1_MINI_E_PANIC_GNR_CURR': ['bin'], 'V1_MINI_E_PANIC_SUBST_CURR': ['bin'],
                                         'V1_MINI_F_AGORA_CURR': ['bin'],
                                         'V1_MINI_F_AGORA_LIFE': ['bin'], 'V1_MINI_G_SOC_CURR': ['bin'],
                                         'V1_MINI_H_SPEC_CURR': ['bin'],
                                         'V1_MINI_I_OBSCOMP_CURR': ['bin'], 'V1_MINI_I_OCD_GNR_CURR': ['bin'],
                                         'V1_MINI_I_OCD_SUBST_CURR': ['bin'],
                                         'V1_MINI_J_PTSTRESS_CURR': ['bin'], 'V1_MINI_K_ALCDEP_12M': ['bin'],
                                         'V1_MINI_K_ALCDEP_LIFE': ['bin'],
                                         'V1_MINI_K_ALCABS_12M': ['bin'], 'V1_MINI_K_ALCABS_LIFE': ['bin'],
                                         'V1_MINI_L_SUBSTDEP_12M': ['bin'],
                                         'V1_MINI_L_SUBSTDEP_LIFE': ['bin'], 'V1_MINI_L_SUBSTABS_12M': ['bin'],
                                         'V1_MINI_L_SUBSTABS_LIFE': ['bin'],
                                         'V1_MINI_M_PSYCH_CURR': ['bin'], 'V1_MINI_M_PSYCH_LIFE': ['bin'],
                                         'V1_MINI_M_MOOD_CURR': ['bin'],
                                         'V1_MINI_M_SCHIZOPH_CURR': ['bin'], 'V1_MINI_M_SCHIZOPH_LIFE': ['bin'],
                                         'V1_MINI_M_SCHIZOAF_CURR': ['bin'],
                                         'V1_MINI_M_SCHIZOAF_LIFE': ['bin'], 'V1_MINI_M_SCHIZOFRM_CURR': ['bin'],
                                         'V1_MINI_M_SCHIZOFRM_LIFE': ['bin'],
                                         'V1_MINI_M_BRIEF_CURR': ['bin'], 'V1_MINI_M_BRIEF_LIFE': ['bin'],
                                         'V1_MINI_M_DELUS_CURR': ['bin'],
                                         'V1_MINI_M_DELUS_LIFE': ['bin'], 'V1_MINI_M_PSYCH_GNR_CURR': ['bin'],
                                         'V1_MINI_M_PSYCH_GNR_LIFE': ['bin'],
                                         'V1_MINI_M_PSYCH_SUBST_CURR': ['bin'], 'V1_MINI_M_PSYCH_SUBST_LIFE': ['bin'],
                                         'V1_MINI_M_NOS_CURR': ['bin'],
                                         'V1_MINI_M_NOS_LIFE': ['bin'], 'V1_MINI_M_MOOD_LIFE': ['bin'],
                                         'V1_MINI_M_MOODNOS_LIFE': ['bin'],
                                         'V1_MINI_M_MAJDEP_CURR': ['bin'], 'V1_MINI_M_MAJDEP_PAST': ['bin'],
                                         'V1_MINI_M_BIPOLI_CURR': ['bin'],
                                         'V1_MINI_M_BIPOLI_PAST': ['bin']},
                                'cytokines': {'IL-6': ['con'], 'IL-7': ['con'],
                                              'IL-8': ['con'], 'IL-10': ['con'], 'IL-12p40': ['con'],
                                              'IL-15': ['con'], 'IL-16': ['con'],
                                              'IL-17': ['con'], 'IL-18': ['con'], 'IL-21': ['con'],
                                              'IL-23': ['con'], 'IL-27': ['con'], 'CRP': ['con'],
                                              'CX3CL1': ['con'], 'GM-CSF': ['con'], 'IFN-g': ['con'],
                                              'SAA': ['con'], 'sICAM-1': ['con'], 'sVCAM-1': ['con'],
                                              'TNF-a': ['con'], 'CXCL10': ['con'], 'CXCL11': ['con'],
                                              'CXCL12': ['con'], 'CCL2': ['con'], 'CCL3': ['con'],
                                              'CCL4': ['con'], 'CCL11': ['con'], 'CCL13': ['con'],
                                              'CCL17': ['con'], 'CCL19': ['con'], 'CCL20': ['con'],
                                              'CCL22': ['con'], 'CCL26': ['con'], 'CCL27': ['con']},
                                'panss': {'V2_PANNS_P1': ['con'], 'V2_PANNS_P2': ['con'], 'V2_PANNS_P3': ['con'],
                                          'V2_PANNS_P4': ['con'], 'V2_PANNS_P5': ['con'], 'V2_PANNS_P6': ['con'],
                                          'V2_PANNS_P7': ['con'], 'V2_PANNS_N1': ['con'], 'V2_PANNS_N2': ['con'],
                                          'V2_PANNS_N3': ['con'], 'V2_PANNS_N4': ['con'], 'V2_PANNS_N5': ['con'],
                                          'V2_PANNS_N6': ['con'], 'V2_PANNS_N7': ['con'], 'V2_PANNS_G1': ['con'],
                                          'V2_PANNS_G2': ['con'], 'V2_PANNS_G3': ['con'], 'V2_PANNS_G4': ['con'],
                                          'V2_PANNS_G5': ['con'], 'V2_PANNS_G6': ['con'], 'V2_PANNS_G7': ['con'],
                                          'V2_PANNS_G8': ['con'], 'V2_PANNS_G9': ['con'], 'V2_PANNS_G10': ['con'],
                                          'V2_PANNS_G11': ['con'], 'V2_PANNS_G12': ['con'], 'V2_PANNS_G13': ['con'],
                                          'V2_PANNS_G14': ['con'], 'V2_PANNS_G15': ['con'], 'V2_PANNS_G16': ['con']},
                                'psp': {'V2_PSP_A': ['con'], 'V2_PSP_B': ['con'], 'V2_PSP_C': ['con'],
                                          'V2_PSP_D': ['con'], 'V2_PSP_SCORE': ['con']},
                                'cgi': {'V2_CGI_SEVERITY': ['con']},
                                'cdss': {'V2_CDSS_1': ['con'], 'V2_CDSS_2': ['con'], 'V2_CDSS_3': ['con'],
                                         'V2_CDSS_4': ['con'], 'V2_CDSS_5': ['con'], 'V2_CDSS_6': ['con'],
                                         'V2_CDSS_7': ['con'], 'V2_CDSS_8': ['con'], 'V2_CDSS_9': ['con']},
                                'swn': {'V2_SWN_1': ['con'], 'V2_SWN_2': ['con'], 'V2_SWN_3': ['con'],
                                        'V2_SWN_4': ['con'], 'V2_SWN_5': ['con'], 'V2_SWN_6': ['con'],
                                        'V2_SWN_7': ['con'], 'V2_SWN_8': ['con'], 'V2_SWN_9': ['con'],
                                        'V2_SWN_10': ['con'], 'V2_SWN_11': ['con'], 'V2_SWN_12': ['con'],
                                        'V2_SWN_13': ['con'], 'V2_SWN_14': ['con'], 'V2_SWN_15': ['con'],
                                        'V2_SWN_16': ['con'], 'V2_SWN_17': ['con'], 'V2_SWN_18': ['con'],
                                        'V2_SWN_19': ['con'], 'V2_SWN_20': ['con']}
                                }


    elif study == 'daniel':
        static_feature_types = {'demographics':{'sexe':['bin'], 'V1_DOV_AGE':['con'],
                                       'V1_RACE':['cat'],'V1_CNTR_RES':['bin'],
                                       'V1_MARRIED':['bin'],'V1_DIVORCED':['bin'],
                                       'V1_OCCUPATION':['bin'],'V1_OCC_TYPE':['cat'],
                                       'V1_OCC_PREV':['bin'],'V1_OCC_PREV_TYPE':['cat'],
                                       'V1_OCC_FATH':['cat'],'V1_OCC_MOTH':['cat'],
                                       'V1_EDUC_YRS':['con'],'V1_EDUC_PAT':['cat'],
                                       'V1_EDUC_FATH':['cat'],'V1_EDUC_MOTH':['cat'],
                                       'V1_LIVING_ALONE':['bin'],'V1_DWELLING':['cat'],
                                       'V1_INCOME_SRC':['cat'],'V1_LIVING_INVIRONMT':['cat']},
                           
                           'diagnosis':{'V1_INC_DSM':['cat'],'V1_INC_PSYCH_DUR':['con'],
                                        'V1_TRSETTING':['cat'],'V1_PSYCHINTV':['bin'],
                                        'V1_PROGNOSIS':['cat'],'V2_HOSP':['bin']},
                          
                           'lifestyle':{'V1_RECDRUGS':['bin'],'V2_RECDRUGS':['bin'],
                                        'V2_CAFFEINE_CUPS':['con'],'V2_LST_CAFFEINE':['cat'],
                                        'V2_ALC_12M':['cat'],'V2_ALC_NEVER':['bin'],
                                        'V2_SMOKE':['cat']},
                          
                           'somatic':{'V1_ECG_NORM':['bin'], 'V1_PE_HEIGHT_CM':['con'],
                                      'V1_PE_WEIGHT_KG':['con'],'V1_PE_WAIST_CM':['con'],
                                      'V1_PE_HIP_CM':['con'],'V1_PE_SBP':['con'],
                                      'V1_PE_DBP':['con'],'V1_PE_PULSE':['con'],
                                      'V2_LSTMEAL':['cat'],'V2_LSTMEAL_TYPE':['cat'],
                                      'BMI':['con']},
                          
                           'treatments':{'avg_dosage':['con']},
                      
                           'cdss':{'V2_CDSS_1':['con'],'V2_CDSS_2':['con'],'V2_CDSS_3':['con'],
                                   'V2_CDSS_4':['con'],'V2_CDSS_5':['con'],'V2_CDSS_6':['con'],
                                   'V2_CDSS_7':['con'],'V2_CDSS_8':['con'],'V2_CDSS_9':['con']},
                          
                           'swn':{'V2_SWN_1':['con'],'V2_SWN_2':['con'],'V2_SWN_3':['con'],
                                  'V2_SWN_4':['con'],'V2_SWN_5':['con'],'V2_SWN_6':['con'],
                                  'V2_SWN_7':['con'],'V2_SWN_8':['con'],'V2_SWN_9':['con'],
                                  'V2_SWN_10':['con'],'V2_SWN_11':['con'],'V2_SWN_12':['con'],
                                  'V2_SWN_13':['con'],'V2_SWN_14':['con'],'V2_SWN_15':['con'],
                                  'V2_SWN_16':['con'],'V2_SWN_17':['con'],'V2_SWN_18':['con'],
                                  'V2_SWN_19':['con'],'V2_SWN_20':['con']},
                          
                           'mini':{'V1_MINI_A_DEPR_CURR':['bin'],'V1_MINI_A_DEPR_RECURR':['bin'],'V1_MINI_A_DPMOOD_GNR_CURR':['bin'],
                                    'V1_MINI_A_DPMOOD_GNR_PAST':['bin'],'V1_MINI_A_DPMOOD_SUBST_CURR':['bin'],'V1_MINI_A_DPMOOD_SUBST_PAST':['bin'],
                                    'V1_MINI_A_MELANCH_CURR':['bin'],'V1_MINI_A_MELANCH_RECURR':['bin'],'V1_MINI_B_DYST_CURR':['bin'],
                                    'V1_MINI_B_DYST_PAST':['bin'],'V1_MINI_C_CURR':['bin'],'V1_MINI_C_RISK':['cat'],
                                    'V1_MINI_D_MANIC_CURR':['bin'],'V1_MINI_D_MANIC_PAST':['bin'],'V1_MINI_D_HYPOM_CURR':['bin'],
                                    'V1_MINI_D_HYPOM_PAST':['bin'],'V1_MINI_D_BIPOLII_CURR':['bin'],'V1_MINI_D_BIPOLII_PAST':['bin'],
                                    'V1_MINI_D_MANIC_GNR_CURR':['bin'],'V1_MINI_D_MANIC_GNR_PAST':['bin'],'V1_MINI_D_MANIC_SUBST_CURR':['bin'],
                                    'V1_MINI_D_MANIC_SUBST_PAST':['bin'],'V1_MINI_E_PANIC_CURR':['bin'],'V1_MINI_E_PANIC_LIFE':['bin'],
                                    'V1_MINI_E_PANIC_GNR_CURR':['bin'],'V1_MINI_E_PANIC_SUBST_CURR':['bin'],'V1_MINI_F_AGORA_CURR':['bin'],
                                    'V1_MINI_F_AGORA_LIFE':['bin'],'V1_MINI_G_SOC_CURR':['bin'],'V1_MINI_H_SPEC_CURR':['bin'],
                                    'V1_MINI_I_OBSCOMP_CURR':['bin'],'V1_MINI_I_OCD_GNR_CURR':['bin'],'V1_MINI_I_OCD_SUBST_CURR':['bin'],
                                    'V1_MINI_J_PTSTRESS_CURR':['bin'],'V1_MINI_K_ALCDEP_12M':['bin'],'V1_MINI_K_ALCDEP_LIFE':['bin'],
                                    'V1_MINI_K_ALCABS_12M':['bin'],'V1_MINI_K_ALCABS_LIFE':['bin'],'V1_MINI_L_SUBSTDEP_12M':['bin'],
                                    'V1_MINI_L_SUBSTDEP_LIFE':['bin'],'V1_MINI_L_SUBSTABS_12M':['bin'],'V1_MINI_L_SUBSTABS_LIFE':['bin'],
                                    'V1_MINI_M_PSYCH_CURR':['bin'],'V1_MINI_M_PSYCH_LIFE':['bin'],'V1_MINI_M_MOOD_CURR':['bin'],
                                    'V1_MINI_M_SCHIZOPH_CURR':['bin'],'V1_MINI_M_SCHIZOPH_LIFE':['bin'],'V1_MINI_M_SCHIZOAF_CURR':['bin'],
                                    'V1_MINI_M_SCHIZOAF_LIFE':['bin'],'V1_MINI_M_SCHIZOFRM_CURR':['bin'],'V1_MINI_M_SCHIZOFRM_LIFE':['bin'],
                                    'V1_MINI_M_BRIEF_CURR':['bin'],'V1_MINI_M_BRIEF_LIFE':['bin'],'V1_MINI_M_DELUS_CURR':['bin'],
                                    'V1_MINI_M_DELUS_LIFE':['bin'],'V1_MINI_M_PSYCH_GNR_CURR':['bin'],'V1_MINI_M_PSYCH_GNR_LIFE':['bin'],
                                    'V1_MINI_M_PSYCH_SUBST_CURR':['bin'],'V1_MINI_M_PSYCH_SUBST_LIFE':['bin'],'V1_MINI_M_NOS_CURR':['bin'],
                                    'V1_MINI_M_NOS_LIFE':['bin'],'V1_MINI_M_MOOD_LIFE':['bin'],'V1_MINI_M_MOODNOS_LIFE':['bin'],
                                    'V1_MINI_M_MAJDEP_CURR':['bin'],'V1_MINI_M_MAJDEP_PAST':['bin'],'V1_MINI_M_BIPOLI_CURR':['bin'],
                                    'V1_MINI_M_BIPOLI_PAST':['bin']}}

    elif study == 'bart':
        # static_feature_types = {'cytokines': {'IL-6': ['con'], 'IL-7': ['con'],
        #                                       'IL-8': ['con'], 'IL-10': ['con'], 'IL-12p40': ['con'],
        #                                       'IL-15': ['con'], 'IL-16': ['con'],
        #                                       'IL-17': ['con'], 'IL-18': ['con'], 'IL-21': ['con'],
        #                                       'IL-23': ['con'], 'IL-27': ['con'], 'CRP': ['con'],
        #                                       'CX3CL1': ['con'], 'GM-CSF': ['con'], 'IFN-g': ['con'],
        #                                       'SAA': ['con'], 'sICAM-1': ['con'], 'sVCAM-1': ['con'],
        #                                       'TNF-a': ['con'], 'CXCL10': ['con'], 'CXCL11': ['con'],
        #                                       'CXCL12': ['con'], 'CCL2': ['con'], 'CCL3': ['con'],
        #                                       'CCL4': ['con'], 'CCL11': ['con'], 'CCL13': ['con'],
        #                                       'CCL17': ['con'], 'CCL19': ['con'], 'CCL20': ['con'],
        #                                       'CCL22': ['con'], 'CCL26': ['con'], 'CCL27': ['con']}}
        static_feature_types = {"cytokines":
                                    {"fr": {"CCL11": ["con"], "CCL13": ["con"], "CCL17": ["con"],
                                            "CCL19": ["con"], "CCL2": ["con"], "CCL20": ["con"],
                                            "CCL22": ["con"], "CCL27": ["con"], "CCL3": ["con"],
                                            "CCL4": ["con"], "CRP": ["con"], "CX3CL1": ["con"],
                                            "CXCL10": ["con"], "CXCL11": ["con"], "CXCL12": ["con"],
                                            "IFN_g": ["con"], "IL_12p40": ["con"], "IL_15": ["con"],
                                            "IL_16": ["con"], "IL_18": ["con"], "IL_21": ["con"],
                                            "IL_23": ["con"], "IL_27": ["con"], "IL_7": ["con"],
                                            "IL_8": ["con"], "SAA": ["con"], "VEGF": ["con"],
                                            "sICAM_1": ["con"], "sVCAM_1": ["con"]},
                                     "fr_missing": {"CCL11": ["con"], "CCL13": ["con"], "CCL17": ["con"],
                                                    "CCL19": ["con"], "CCL2": ["con"], "CCL20": ["con"],
                                                    "CCL22": ["con"], "CCL26": ["con"], "CCL27": ["con"],
                                                    "CCL3": ["con"], "CCL4": ["con"], "CRP": ["con"],
                                                    "CX3CL1": ["con"], "CXCL10": ["con"], "CXCL11": ["con"],
                                                    "CXCL12": ["con"], "GM_CSF": ["con"], "IFN_g": ["con"],
                                                    "IL_10": ["con"], "IL_12p40": ["con"], "IL_12p70": ["con"],
                                                    "IL_13": ["con"], "IL_15": ["con"], "IL_16": ["con"],
                                                    "IL_17": ["con"], "IL_18": ["con"], "IL_1a": ["con"],
                                                    "IL_1b": ["con"], "IL_2": ["con"], "IL_21": ["con"],
                                                    "IL_23": ["con"], "IL_27": ["con"], "IL_4": ["con"],
                                                    "IL_5": ["con"], "IL_6": ["con"], "IL_7": ["con"],
                                                    "IL_8": ["con"], "SAA": ["con"], "TNF_a": ["con"],
                                                    "TNF_b": ["con"], "VEGF": ["con"], "sICAM_1": ["con"],
                                                    "sVCAM_1": ["con"]},
                                     "umcu": {"CCL11": ["con"], "CCL13": ["con"], "CCL17": ["con"], "CCL19": ["con"],
                                              "CCL2": ["con"], "CCL20": ["con"], "CCL22": ["con"], "CCL26": ["con"],
                                              "CCL27": ["con"], "CCL3": ["con"], "CCL4": ["con"], "CRP": ["con"],
                                              "CX3CL1": ["con"], "CXCL10": ["con"], "CXCL11": ["con"],
                                              "CXCL12": ["con"], "GM-CSF": ["con"], "IFN-g": ["con"], "IL-10": ["con"],
                                              "IL-12p40": ["con"], "IL-13": ["con"], "IL-15": ["con"],
                                              "IL-16": ["con"], "IL-17": ["con"], "IL-18": ["con"], "IL-21": ["con"],
                                              "IL-23": ["con"], "IL-27": ["con"], "IL-6": ["con"], "IL-7": ["con"],
                                              "IL-8": ["con"], "SAA": ["con"], "sICAM-1": ["con"], "sVCAM-1": ["con"],
                                              "TNF-a": ["con"], "TNF-b": ["con"], "VEGF": ["con"]}}}

    with open(data_path + study + '_static_features.json', 'w') as file_object:
        json.dump(static_feature_types, file_object, ensure_ascii=False,indent=4)

    return



def feature_extraction_dense(static_features, static_feature_types, 
                        nan_strategy='promise', impute_strategy='KNN'):
    
    #nan_strategy='remove_samples', 'remove_features', 'impute', None
    
    if nan_strategy == 'remove_samples':
        aggregation='intersection'
        for k in static_features.keys():
             static_features[k] = static_features[k].dropna()
    elif nan_strategy == 'remove_features':
        aggregation='intersection'
        keys = list(static_features.keys())
        for k in keys:
            temp = static_features[k].dropna(how='all')
            temp = temp.dropna(axis=1)
            if temp.shape[1]>0:
                static_features[k] = temp
            else:
                static_features.pop(k)
    else:    
        aggregation='union'
    
    subjects = []
    for k in static_features.keys():
        subjects.append(list(static_features[k].index))
    if aggregation == 'intersection':
        s = list(set(subjects[0]).intersection(*subjects))
    elif aggregation == 'union':
        s = list(set(subjects[0]).union(*subjects))
    s.sort()
    
    static_features1 = copy.deepcopy(static_features)
    n = [0]
    for k in static_features.keys():
        n.append(static_features[k].shape[1])
        static_features1[k] = static_features1[k].reindex(s)
    n = np.cumsum(n)
    if nan_strategy == 'impute':
        temp = [static_features1[k] for k in static_features.keys()]
        temp = np.concatenate(temp, axis=1)
        if impute_strategy == 'KNN': 
            imp = KNNImputer(missing_values=np.nan, n_neighbors=1)
        elif impute_strategy == 'zero':
            imp =  SimpleImputer(missing_values=np.nan, strategy='constant')
        temp = imp.fit_transform(temp)
        for k,key in enumerate(static_features.keys()):
            static_features1[key] = pd.DataFrame(temp[:,n[k]:n[k+1]], 
                                                 columns=static_features[key].columns, 
                                                 index=s)

    features = {}
    for k in static_features.keys():
        d = static_features1[k]
        
        temp = []
        for c, col in enumerate(d.columns):
            if static_feature_types[k][col][0] == 'bin':
                d[col].iloc[np.where(d[col] == 2)] = -1 #0
                #if nan_strategy == 'impute':
                #    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                #    d[col] = imp.fit_transform(d[col].values.reshape(-1, 1))
                temp.append(d[col].values.reshape(-1, 1))
                static_feature_types[k][col] = ['bin', 2]
            elif static_feature_types[k][col][0] == 'cat':
                #if nan_strategy == 'impute':
                #    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                #    d[col] = imp.fit_transform(d[col].values.reshape(-1, 1))
                min_cat = int(min(1,d[col].min()))
                ohe = OneHotEncoder(sparse=False, categories=[
                    list(range(min_cat, int(d[col].max() + 1)))])  # , categories=list(range(int(d[col].max()+1)))
                data = d[col].values
                non_nan_index = np.isfinite(data)
                non_nan_data = data[non_nan_index].reshape(-1, 1)
                ohe = ohe.fit(non_nan_data)
                temp_data = np.zeros([data.shape[0], len(ohe.categories[0])])
                temp_data[non_nan_index,:] = ohe.transform(non_nan_data)
                temp_data[np.logical_not(non_nan_index),:] = np.nan
                temp.append(temp_data)
                static_feature_types[k][col] = ['cat', temp[c].shape[1]]
            elif static_feature_types[k][col][0] == 'con':
                #if nan_strategy == 'impute':
                #    imp =  SimpleImputer(missing_values=np.nan, strategy='median')
                #    d[col] = imp.fit_transform(d[col].values.reshape(-1, 1))
                data = d[col].values
                non_nan_index = np.isfinite(data)
                non_nan_data = data[non_nan_index].reshape(-1, 1)
                sclr = scaler(scaler_type='standardize')
                sclr.fit(non_nan_data)
                temp_data = np.zeros([data.shape[0], 1])
                temp_data[non_nan_index,:] = sclr.transform(non_nan_data)
                temp_data[np.logical_not(non_nan_index),:] = np.nan
                temp.append(temp_data)
                static_feature_types[k][col] = ['con', 1]
        features[k] = pd.DataFrame(np.concatenate(temp, axis=1), index=d.index)

    return features, static_feature_types


def swn_transform(static_features, static_feature_types):
    
    swn = copy.deepcopy(static_features['swn']).to_numpy()
    transposed_items = [0, 3, 5, 8, 9, 10, 11, 13, 15, 16]
    for item in transposed_items:
        swn[:, item] = 7 - swn[:, item]
    swn_features = np.zeros([swn.shape[0], 6])
    swn_features[:,0] = swn[:,2] + swn[:,6] + swn[:,10] + swn[:,16]
    swn_features[:,1] = swn[:,0] + swn[:,11] + swn[:,14] + swn[:,18]
    swn_features[:,2] = swn[:,3] + swn[:,9] + swn[:,17] + swn[:,19]
    swn_features[:,3] = swn[:,1] + swn[:,4] + swn[:,8] + swn[:,15]
    swn_features[:,4] = swn[:,5] + swn[:,7] + swn[:,12] + swn[:,13]
    swn_features[:,5] = np.sum(swn, axis=1)
    
    static_features['swn'] = pd.DataFrame(swn_features, columns=['mental_func',
                                                                 'self_control',
                                                                 'emotional_regu',
                                                                 'physical_func',
                                                                 'social_integ',
                                                                 'total_swn'],
                                          index=static_features['swn'].index)
    static_feature_types['swn'] = {'mental_func':['con',1], 'self_control':['con',1],
                                   'emotional_regu':['con',1], 'physical_func':['con',1],
                                   'social_integ':['con',1], 'total_swn':['con',1]}
    
    return static_features, static_feature_types