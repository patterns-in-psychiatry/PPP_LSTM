#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:54:02 2020

@author: seykia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns        


def plot_classification_results(inputs, auc, std, classifier_names):

    input_num = len(inputs)
    classifier_num = auc.shape[0]
    
    ind = np.arange(input_num)  # the x locations for the groups
    
    width = 1/classifier_num - 0.1/classifier_num   # the width of the bars
    
    fig, ax = plt.subplots()
    for i in range(classifier_num):
        ax.bar(ind + (i - np.floor(classifier_num/2)) * width, auc[i,:], width, 
               yerr=std[i,:], label=classifier_names[i])
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AUC')
    ax.set_title('Classifier and Measure Comparison')
    ax.set_xticks(ind)
    ax.set_xticklabels(inputs)
    ax.set_ylim([0.25,0.85])
    ax.legend(loc=8, ncol = 3, bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    
def plot_classification_results2(aucs, classifier_names, save_path=None):

    classifier_num = aucs.shape[1]
            
    fig, ax = plt.subplots(figsize=[18,8])
    ax.bar(np.arange(classifier_num), np.mean(aucs, axis=0), yerr=np.std(aucs, axis=0))
    ax.set_ylabel('AUC')
    ax.set_title('Classifier Comparison')
    ax.set_xticks(np.arange(classifier_num))
    ax.set_xticklabels(classifier_names)
    if save_path is not None:
        fig.savefig(save_path + '.png', dpi=150)
    
    

def plot_regression_results(inputs, mae, std, regressor_names):
    
    input_num = len(inputs)
    regressor_num = mae.shape[0]
    
    ind = np.arange(input_num)  # the x locations for the groups
    
    width = 1/regressor_num - 0.1/regressor_num   # the width of the bars
    
    fig, ax = plt.subplots()
    for i in range(regressor_num):
        ax.bar(ind + (i - np.floor(regressor_num/2)) * width, mae[i,:], width, 
               yerr=std[i,:], label=regressor_names[i])
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MAE')
    ax.set_title('Regressor and Measure Comparison')
    ax.set_xticks(ind)
    ax.set_xticklabels(inputs)
    ax.set_ylim([mae.min()-std.max(),mae.max()+std.max()])
    ax.legend(loc=8, ncol = 3, bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    

def plot_regression_results2(ax, y_true, y_pred, title, scores):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    ax.set_title(title)


def plot_results(prediction_summary, acc, prognosis=None, save_path=None):
    
    categories = ['DR', 'PR', 'UR', 'US', 'UN', 'PN', 'DN']
    
    if prognosis is None:
        pie_plt_num = 2
        bar_plt_num = 1
    else:
        pie_plt_num = 3
        bar_plt_num = 2
   
    plt.figure(figsize=[18,8])
    ax = plt.subplot(1,pie_plt_num,1)
    ax.axis('equal')
    groups = ['R', 'N']
    numbers = [np.sum(prediction_summary['PANNS_Label']), prediction_summary.shape[0]-np.sum(prediction_summary['PANNS_Label'])]
    ax.pie(numbers, labels = groups, autopct='%1.2f%%', explode=[0.05]*len(groups), pctdistance=0.5)
    ax.set_title('Data Distribution')
    
    ax = plt.subplot(1,pie_plt_num,2)
    ax.axis('equal')
    numbers = [np.sum(prediction_summary['PANNS_Recom']==cat) for cat in categories]    
    ax.pie(numbers, labels = categories, autopct='%1.2f%%', explode=[0.05]*len(categories), pctdistance=0.5)
    ax.set_title('Model')
    
    if pie_plt_num == 3:
        ax = plt.subplot(1,pie_plt_num,3)
        ax.axis('equal')
        numbers = [np.sum(prognosis['PANNS_Recom']==cat) for cat in categories]    
        ax.pie(numbers, labels = categories, autopct='%1.2f%%', explode=[0.05]*len(categories), pctdistance=0.5)
        ax.set_title('Expert')
    
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    
    if save_path is not None:
        fig1.savefig(save_path + 'Pie_Chart.png', dpi=150)
    
    categories = ['DR', 'PR', 'UR', 'UN', 'PN', 'DN']
    
    
    plt.figure(figsize=[18,8])        
    x = np.arange(len(categories))
    ax = plt.subplot(1,bar_plt_num,1)
    temp = list(acc.values())
    temp = np.array(temp)
    temp[temp==0] = 0.001
    temp[np.isnan(temp)] = 0.001
    ax.bar(x, temp.squeeze(), align='center', alpha=0.5, ecolor='black')
    ax.set_title('Model Performance')
    ax.yaxis.grid(True)
    plt.xticks(x, categories)
    plt.ylim([0,1])
    
    if bar_plt_num==2:
        acc_doc = dict()  
        for i,c in enumerate(categories):
            a = prognosis.loc[prognosis['PANNS_Recom']==c].iloc[:,0].to_numpy() 
            if c in ['DR', 'PR', 'UR']:
                acc_doc[c] = np.sum(a)/a.shape[0]
            elif c in ['DN', 'PN', 'UN']:
                acc_doc[c] = np.sum(a==0)/a.shape[0]
        ax = plt.subplot(1,bar_plt_num,2)
        temp = list(acc_doc.values())
        temp = np.array(temp)
        temp[temp==0] = 0.001
        temp[np.isnan(temp)] = 0.001
        ax.bar(x, temp.squeeze(), align='center', alpha=0.5, ecolor='black')
        ax.set_title('Expert Performance')
        ax.yaxis.grid(True)
        plt.xticks(x, categories)
        plt.ylim([0,1])
    
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    if save_path is not None:
        fig1.savefig(save_path + 'Accuracies_Bar_Chart.png', dpi=150)
    
    
def plot_CFI_group_effect(cfi, save_path=None):  
    
    sfts = list(cfi[list(cfi.keys())[0]].keys())
    
    for sft in sfts:
        effect_size = dict()
        features = cfi[list(cfi.keys())[0]][sft].keys()
        for feature in features:
            effect_size[feature] = np.zeros([len(cfi.keys()),])
            for s,subject in enumerate(cfi.keys()):
                effect_size[feature][s] = np.abs(cfi[subject][sft][feature]['effect_size'])
                
        mean_effect_size = dict()        
        for feature in effect_size.keys():
            mean_effect_size[feature] = np.mean(effect_size[feature])
                
        df = pd.DataFrame(effect_size)
        
        plt.figure(figsize=[18,8])       
        sns.boxplot(data=df, orient="v", palette="Set2")
        plt.xticks(rotation=90)
        plt.title(sft)
        plt.tight_layout()
        fig1 = plt.gcf()
        plt.show()
        
        if save_path is not None:
            fig1.savefig(save_path + 'Group_Feature_Importance_' + sft + '.png', dpi=150)
    