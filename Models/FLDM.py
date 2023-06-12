#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:12:47 2021

@author: seykia
"""

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix


class FLDM:
    
    def __init__(self, std=0.15):
        
        self.x = np.arange(0, 1.01, 0.01)
        self.verylow_pr = fuzz.gaussmf(self.x, 0, std)
        self.low_pr = fuzz.gaussmf(self.x, 0.25, std/2)
        self.medium_pr = fuzz.gaussmf(self.x, 0.5,std)
        self.high_pr = fuzz.gaussmf(self.x, 0.75,std/2)
        self.veryhigh_pr = fuzz.gaussmf(self.x, 1,std)
        
        self.def_r =  fuzz.gaussmf(self.x, 1,std/2)
        self.prob_r =  fuzz.gaussmf(self.x, 0.80,std/2)
        self.unsure_r =  fuzz.gaussmf(self.x, 0.6, std/2)
        self.unsure =  fuzz.gaussmf(self.x, 0.5, std/4)
        self.unsure_n =  fuzz.gaussmf(self.x, 0.4, std/2)
        self.prob_n =  fuzz.gaussmf(self.x, 0.2, std/2)
        self.def_n =  fuzz.gaussmf(self.x, 0, std/2)
    
    def predict(self, p, wp, bp, plot=False):

        mf_verylow_pr = fuzz.interp_membership(self.x, self.verylow_pr, p)        
        mf_low_pr = fuzz.interp_membership(self.x, self.low_pr, p)
        mf_medium_pr = fuzz.interp_membership(self.x, self.medium_pr, p)
        mf_high_pr = fuzz.interp_membership(self.x, self.high_pr, p)
        mf_veryhigh_pr = fuzz.interp_membership(self.x, self.veryhigh_pr, p)
        
        mf_verylow_wpr = fuzz.interp_membership(self.x, self.verylow_pr, wp)
        mf_low_wpr = fuzz.interp_membership(self.x, self.low_pr, wp)
        mf_medium_wpr = fuzz.interp_membership(self.x, self.medium_pr, wp)
        mf_high_wpr = fuzz.interp_membership(self.x, self.high_pr, wp)
        mf_veryhigh_wpr = fuzz.interp_membership(self.x, self.veryhigh_pr, wp)
        
        mf_verylow_bpr = fuzz.interp_membership(self.x, self.verylow_pr, bp)
        mf_low_bpr = fuzz.interp_membership(self.x, self.low_pr, bp)
        mf_medium_bpr = fuzz.interp_membership(self.x, self.medium_pr, bp)
        mf_high_bpr = fuzz.interp_membership(self.x, self.high_pr, bp)
        mf_veryhigh_bpr = fuzz.interp_membership(self.x, self.veryhigh_pr, bp)
        
        DR_rule = np.fmin(np.fmin(np.fmax(mf_high_wpr, mf_veryhigh_wpr), mf_veryhigh_pr), self.def_r)
        PR_rule = np.fmin(np.fmin(np.fmax(mf_high_pr, mf_veryhigh_pr), mf_medium_wpr), self.prob_r)
        UR_rule = np.fmin(np.fmin(np.fmax(mf_high_pr, mf_veryhigh_pr), np.fmax(mf_low_wpr, mf_verylow_wpr)), self.unsure_r)
        US_rule = np.fmin(np.fmin(mf_medium_pr, np.fmin(mf_verylow_wpr, mf_veryhigh_bpr)), self.unsure)
        UN_rule = np.fmin(np.fmin(np.fmax(mf_low_pr, mf_verylow_pr), np.fmax(mf_high_bpr, mf_veryhigh_bpr)), self.unsure_n)
        PN_rule = np.fmin(np.fmin(np.fmax(mf_low_pr, mf_verylow_pr), mf_medium_bpr), self.prob_n)
        DN_rule = np.fmin(np.fmin(np.fmax(mf_low_bpr,mf_verylow_bpr),  mf_verylow_pr), self.def_n)
        
        aggregated = np.fmax(DR_rule, np.fmax(PR_rule, np.fmax(UR_rule, np.fmax(US_rule, np.fmax(UN_rule, np.fmax(PN_rule, DN_rule))))))
        
        pr = fuzz.defuzz(self.x, aggregated, 'centroid')
       
        a = dict()
        a['DR'] = fuzz.interp_membership(self.x, DR_rule, pr)  
        a['PR'] = fuzz.interp_membership(self.x, PR_rule, pr)  
        a['UR'] = fuzz.interp_membership(self.x, UR_rule, pr) + np.finfo(float).eps
        a['US'] = fuzz.interp_membership(self.x, US_rule, pr) 
        a['UN'] = fuzz.interp_membership(self.x, UN_rule, pr) + np.finfo(float).eps
        a['PN'] = fuzz.interp_membership(self.x, PN_rule, pr)  
        a['DN'] = fuzz.interp_membership(self.x, DN_rule, pr)  
    
        decision = max(a, key=a.get)
        
        
        if plot:
            pr_activation = fuzz.interp_membership(self.x, aggregated, pr)  # for plot
            fig, ax0 = plt.subplots(figsize=(8, 3))
            pr0 = np.zeros_like(self.x)
            ax0.fill_between(self.x, pr0, DR_rule, facecolor='b', alpha=0.5)
            ax0.plot(self.x, self.def_r, 'b', linewidth=0.5, linestyle='--', label='DR')
            ax0.fill_between(self.x, pr0, PR_rule, facecolor='g', alpha=0.5)
            ax0.plot(self.x, self.prob_r, 'g', linewidth=0.5, linestyle='--', label='PR')
            ax0.fill_between(self.x, pr0, US_rule, facecolor='r', alpha=0.5)
            ax0.plot(self.x, self.unsure, 'r', linewidth=0.5, linestyle='--', label='US')
            ax0.fill_between(self.x, pr0, UR_rule, facecolor='k', alpha=0.5)
            ax0.plot(self.x, self.unsure_r, 'k', linewidth=0.5, linestyle='--', label='UR')
            ax0.fill_between(self.x, pr0, UN_rule, facecolor='m', alpha=0.5)
            ax0.plot(self.x, self.unsure_n, 'm', linewidth=0.5, linestyle='--', label='UN')
            ax0.fill_between(self.x, pr0, PN_rule, facecolor='y', alpha=0.5)
            ax0.plot(self.x, self.prob_n, 'y', linewidth=0.5, linestyle='--', label='PN')
            ax0.fill_between(self.x, pr0, DN_rule, facecolor='c', alpha=0.5)
            ax0.plot(self.x, self.def_n, 'c', linewidth=0.5, linestyle='--', label='DN')
            ax0.plot([pr, pr], [0, pr_activation], 'k', linewidth=1.5, alpha=0.9)
            ax0.set_title('IPR=%.2f, WCPR=%.2f, BCPR=%.2f => FPR=%.2f' %(p,wp,bp,pr))
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            
            #fig, ax0 = plt.subplots(figsize=(8, 3))
            #ax0.plot(self.x, self.def_r, 'b', linewidth=0.5, linestyle='--', )
            #ax0.plot(self.x, self.prob_r, 'g', linewidth=0.5, linestyle='--')
            #ax0.plot(self.x, self.unsure, 'r', linewidth=0.5, linestyle='--')
            #ax0.plot(self.x, self.prob_n, 'y', linewidth=0.5, linestyle='--')
            #ax0.plot(self.x, self.def_n, 'c', linewidth=0.5, linestyle='--')
            #ax0.fill_between(self.x, pr0, aggregated, facecolor='Orange', alpha=0.7)
            #ax0.plot([pr, pr], [0, pr_activation], 'k', linewidth=1.5, alpha=0.9)
            #ax0.set_title('Final Probaility of Remission.')

        return pr, decision, a

    def plot_mfs(self):
        """
        Plotting membership functions.
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex='all',
                                                     sharey='all')
        ax0.plot(self.x, self.verylow_pr, linewidth=2.5, label='Very Low', alpha=0.8)
        ax0.plot(self.x, self.low_pr, linewidth=2.5, label='Low', alpha=0.8)
        ax0.plot(self.x, self.medium_pr, linewidth=2.5, label='Medium', alpha=0.8)
        ax0.plot(self.x, self.high_pr, linewidth=2.5, label='High', alpha=0.8)
        ax0.plot(self.x, self.veryhigh_pr, linewidth=2.5, label='Very High', alpha=0.8)
        for spine in ax0.spines.values():
            spine.set_visible(False)
        ax0.grid(linestyle = '--', linewidth = 0.5)
        ax0.set_title('Predicted Probability')
        ax0.legend(fontsize=14)
        ax0.tick_params(axis='both', which='major', labelsize=14)
        
        ax1.plot(self.x, self.def_r, linewidth=2.5, label='DR', alpha=0.8)
        ax1.plot(self.x, self.prob_r, linewidth=2.5, label='PR', alpha=0.8)
        ax1.plot(self.x, self.unsure_r, linewidth=2.5, label='UR', alpha=0.8)
        ax1.plot(self.x, self.unsure, linewidth=2.5, label='US', alpha=0.8)
        ax1.plot(self.x, self.unsure_n,linewidth=2.5, label='UN', alpha=0.8)
        ax1.plot(self.x, self.prob_n, linewidth=2.5, label='PN', alpha=0.8)
        ax1.plot(self.x, self.def_n, linewidth=2.5, label='DN', alpha=0.8)
        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.grid(linestyle = '--', linewidth = 0.5)
        ax1.set_title('Decisions')
        ax1.legend(fontsize=14,  ncol=2, loc="lower right")
        ax1.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()

    
    def plot_decision_surface(self):
        
        fontsize=14
        prob = np.arange(0,1.01,0.01)
        uncr = np.arange(0.01,1.01,0.01)
        pred = np.zeros([prob.shape[0], uncr.shape[0]])
        dec = np.zeros([prob.shape[0], uncr.shape[0]])
        decisions = ['DN', 'PN', 'UN', 'US', 'UR', 'PR', 'DR']
        for i in range(prob.shape[0]):
            for j in range(uncr.shape[0]):
                pr, d, a = self.predict(prob[i], max(0, prob[i]-uncr[j]), min(1, prob[i]+uncr[j]))
                pred[i,j] = pr
                dec[i,j] = decisions.index(d)               
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        im0 = ax0.imshow(np.flip(pred, axis=0), vmin=0, vmax=1, cmap='RdBu_r')
        im1 = ax1.imshow(np.flip(dec, axis=0), cmap='RdBu_r')
        for spine in ax0.spines.values():
            spine.set_visible(False)
        ax0.set_title('Uncertainty-Adjusted Predicted Probability', fontsize=fontsize)
        fig.colorbar(im0, ax=ax0, shrink=0.9)
        ax0.set_xticks(np.arange(0,120,20))
        ax0.set_xticklabels(['0','0.2','0.4','0.6','0.8','1'], fontsize=fontsize)
        ax0.set_yticks(np.arange(0,120,20))
        ax0.set_yticklabels(['1','0.8','0.6','0.4','0.2','0'], fontsize=fontsize)
        ax0.set_ylabel('Predicted Probability', fontsize=fontsize)
        ax0.set_xlabel('Uncertainty', fontsize=fontsize)
        for spine in ax1.spines.values():
            spine.set_visible(False)        
        ax1.set_title('Decisions', fontsize=fontsize)
        cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.9)
        cbar1.ax.set_yticklabels(decisions, fontsize=fontsize)
        ax1.set_xticks(np.arange(0,120,20))
        ax1.set_xticklabels(['0','0.2','0.4','0.6','0.8','1'], fontsize=fontsize)
        ax1.set_yticks(np.arange(0,120,20))
        ax1.set_yticklabels(['1','0.8','0.6','0.4','0.2','0'], fontsize=fontsize)
        ax1.set_ylabel('Predicted Probability', fontsize=fontsize)
        ax1.set_xlabel('Uncertainty', fontsize=fontsize)
        
    def test(self):
        
        print('Press Ctrl+c to stop the test...')
        i=0
        while True:
            temp = np.sort(np.random.randint(0,100,3))/100
            _ = self.predict(temp[1],temp[0],temp[2])
            i += 1
            if i%1000==0:
                print('Success after %d tests. Press Ctrl+c to stop the test...' %(i))
        print('Failed at %d test, for pr=%f, wpr=%f, bpr=%f' %(i,temp[1],temp[0],temp[2]))
    
    
    def multi_class_fit_transform(self, pr, wpr, bpr):
    
        mpr = np.zeros([pr.shape[0], pr.shape[1]])
        for i in range(pr.shape[0]):
            for j in range(pr.shape[1]):
                #fdm = FLDM()  
                mpr[i,j],b,c = self.predict(pr[i,j], wpr[i,j], bpr[i,j])
        for i in range(pr.shape[0]):
            mpr[i,:] = mpr[i,:]/np.sum(mpr[i,:])
            
        return mpr
        

def decision_making(prediction_summary, method='fuzzy', criteria=['PANNS'], tr=0.5, plot=True):
    
    if method == 'fuzzy':
        for c in criteria:     
            pr = np.zeros([prediction_summary.shape[0],])
            for i, idx in enumerate(list(prediction_summary.index)):
                fdm = FLDM()  
                pr[i],b,_ = fdm.predict(prediction_summary.loc[idx, c+'_Prob'], 
                                    prediction_summary.loc[idx, c+'_LB'], 
                                    prediction_summary.loc[idx,c+'_UB'])
                prediction_summary.loc[idx, c+'_Recom'] = b
            prediction_summary[c+'_FDM'] = pr
            
    elif method == 'plain':
        for c in criteria:
            prediction_summary.loc[prediction_summary.loc[prediction_summary.loc[:,c+'_Prob'].to_numpy() >= tr].index,c+'_Recom'] = 'PR' 
            prediction_summary.loc[prediction_summary.loc[prediction_summary.loc[:,c+'_Prob'].to_numpy() < tr].index,c+'_Recom'] = 'PN'
             
            prediction_summary.loc[prediction_summary.loc[prediction_summary.loc[:,c+'_LB'].to_numpy() >= 0.75].index,c+'_Recom'] = 'DR'
            prediction_summary.loc[prediction_summary.loc[prediction_summary.loc[:,c+'_UB'].to_numpy() <= 0.25].index,c+'_Recom'] = 'DN' 
            
            prediction_summary.loc[prediction_summary.loc[np.logical_and(prediction_summary.loc[:,c+'_Prob'].to_numpy() >= tr,
                                       prediction_summary.loc[:,c+'_UB'].to_numpy() - prediction_summary.loc[:,c+'_LB'].to_numpy() >= 0.5)].index,c+'_Recom'] = 'UR' 
            prediction_summary.loc[prediction_summary.loc[np.logical_and(prediction_summary.loc[:,c+'_Prob'].to_numpy() < tr,
                                       prediction_summary.loc[:,c+'_UB'].to_numpy() - prediction_summary.loc[:,c+'_LB'].to_numpy() >= 0.5)].index,c+'_Recom'] = 'UN' 
            
            prediction_summary.loc[prediction_summary.loc[np.logical_and(np.logical_and(prediction_summary.loc[:,c+'_Prob'].to_numpy() < tr+0.05,
                                                                                        prediction_summary.loc[:,c+'_Prob'].to_numpy() > tr-0.05),
                                       prediction_summary.loc[:,c+'_UB'].to_numpy() - prediction_summary.loc[:,c+'_LB'].to_numpy() >= 0.5)].index,c+'_Recom'] = 'US' 
    elif method == 'threshold': 
        for c in criteria:
            prediction_summary.loc[prediction_summary[c+'_Prob']<0.5,c+'_Recom'] = 'DN'   
            prediction_summary.loc[prediction_summary[c+'_Prob']>=0.5,c+'_Recom'] = 'DR'   
                  
    acc = dict()  
    for i,c in enumerate(['DR', 'PR', 'UR', 'UN', 'PN', 'DN']):
        a = prediction_summary.loc[prediction_summary['PANNS_Recom']==c,'PANNS_Label'].to_numpy() 
        if a.shape[0] == 0:
            acc[c] = 0
            continue
        if c in ['DR', 'PR', 'UR']:
            acc[c] = np.sum(a)/a.shape[0]
        elif c in ['DN', 'PN', 'UN']:
            acc[c] = np.sum(a==0)/a.shape[0]
    
    return prediction_summary, acc
        


def evaluate_decision_making(prediction_summary, positive_categories = ['DR', 'PR', 'UR'], 
          negative_categories = ['UN', 'PN', 'DN'], verbose=0):
    
    results = dict()
    undecided = list(set(['DR', 'PR', 'UR', 'US', 'UN', 'PN', 'DN']) - set(positive_categories) - set(negative_categories))
    a = prediction_summary.loc[prediction_summary['PANNS_Recom'].isin(positive_categories+negative_categories)]
    b = prediction_summary.loc[prediction_summary['PANNS_Recom'].isin(undecided)]
    
    results['decisiveness'] = a.shape[0]/prediction_summary.shape[0]
    a.loc[a['PANNS_Recom'].isin(positive_categories),'prediction'] = 1.0
    a.loc[a['PANNS_Recom'].isin(negative_categories),'prediction'] = 0.0
    cf = confusion_matrix(a['PANNS_Label'].to_numpy(np.float), 
                          a['prediction'].to_numpy(), labels=[0,1])
    TN1 = cf[0][0]
    FN1 = cf[1][0]
    TP1 = cf[1][1]
    FP1 = cf[0][1]
    results['risk'] = (FP1 + FN1)/prediction_summary.shape[0]
    results['sensitivity'] = TP1 / (TP1 + FN1)
    results['specificity'] = TN1 / (TN1 + FP1)
    
    try:
        b.loc[b['PANNS_Prob']>=0.5,'prediction'] = 1.0
        b.loc[b['PANNS_Prob']<0.5,'prediction'] = 0.0
        cf = confusion_matrix(b['PANNS_Label'].to_numpy(np.float), 
                              b['prediction'].to_numpy(), labels=[0,1])
        TN2 = cf[0][0]
        FN2 = cf[1][0]
        TP2 = cf[1][1]
        FP2 = cf[0][1]
    except:
        FP2 = 0
        FN2 = 0
    results['gain'] = (FP2 + FN2)/(FP1 + FN1 + FP2 + FN2)

    if verbose!=0:
        print('Desisiveness=%f, Risk=%f, Gain=%f' %(results['decisiveness'],
                                                results['risk'],results['gain']))
    
    return results



def risk_gain(prediction_summary, trs = np.arange(0.1,1,0.01)):
    
    results = dict()
    trs = np.arange(0.1,1,0.01)
    results['decisiveness'] = np.zeros([trs.shape[0]])
    results['risk'] = np.zeros([trs.shape[0]])
    results['sensitivity'] = np.zeros([trs.shape[0]])
    results['specificity'] = np.zeros([trs.shape[0]])
    results['gain'] = np.zeros([trs.shape[0]])
    
    for t, tr in enumerate(trs):
        idx = np.logical_or(prediction_summary['PANNS_FDM']<=tr/2, 
                             prediction_summary['PANNS_FDM']>=1-tr/2)
        a = prediction_summary.loc[idx]
        b = prediction_summary.loc[np.logical_not(idx)]
        
        results['decisiveness'][t] = a.shape[0]/prediction_summary.shape[0]
        
        try:
            a.loc[a['PANNS_FDM']<=tr/2, 'prediction'] = 0.0
            a.loc[a['PANNS_FDM']>=1-tr/2, 'prediction'] = 1.0
            cf = confusion_matrix(a['PANNS_Label'].to_numpy(np.float), 
                                  a['prediction'].to_numpy(), labels=[0,1])
            TN1 = cf[0][0]
            FN1 = cf[1][0]
            TP1 = cf[1][1]
            FP1 = cf[0][1]
        except:
            TN1 = np.nan
            FN1 =  np.nan
            TP1 =  np.nan
            FP1 =  np.nan
        results['risk'][t] = (FP1 + FN1)/prediction_summary.shape[0]
        results['sensitivity'][t] = TP1 / (TP1 + FN1)
        results['specificity'][t] = TN1 / (TN1 + FP1)
        
        try:
            b.loc[b['PANNS_Prob']>=0.5,'prediction'] = 1.0
            b.loc[b['PANNS_Prob']<0.5,'prediction'] = 0.0
            cf = confusion_matrix(b['PANNS_Label'].to_numpy(np.float), 
                                  b['prediction'].to_numpy(), labels=[0,1])
            TN2 = cf[0][0]
            FN2 = cf[1][0]
            TP2 = cf[1][1]
            FP2 = cf[0][1]
        except:
            FP2 = 0
            FN2 = 0
        results['gain'][t] = (FP2 + FN2)/(FP1 + FN1 + FP2 + FN2)
    
    return results


    