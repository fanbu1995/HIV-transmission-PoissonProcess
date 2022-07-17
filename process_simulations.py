#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 20:54:48 2021

@author: fan
"""

#%%
# process simulation results 
import os
os.chdir('/Users/fan/Documents/Research_and_References/HIV_transmission_flow/')

from copy import copy#, deepcopy

import matplotlib.pyplot as plt

#from utilsHupdate import *

import numpy as np

import pandas as pd
#from copy import copy

# 10/11/2020 update
# numpy.random new generator...
from numpy.random import default_rng
rng = default_rng()

import pickle as pkl


#%%

# plot the posterior means of weightMFs (the weights on the largest two components: 25 vs 35)

def poolWeights(res_dir, nums, last=500, savepath=None):
    weight_means = np.zeros((len(nums), 2))
    for i in range(len(nums)):
        n = nums[i]
        fpath = res_dir + 'weightMF_'+str(n)+'.pkl'
        W = pkl.load(file=open(fpath,'rb'))
        chain = np.array(W[-last:])[:,:2]
        w_means = np.mean(chain, axis=0)
        weight_means[i,:] = w_means
        
    if max(nums) < 200:
        # put weight on younger men to the left side
        weight_means = weight_means[:,::-1]
        
    data = [weight_means[:,0], weight_means[:,1]]
    
    plt.figure(figsize=(6,4))
        
#    bplot = plt.boxplot(data, vert = True, patch_artist=True,
#                labels=['younger','older'])
    
    vplot = plt.violinplot(data, showmeans=True, widths=0.5)
    
    plt.grid(axis='y')
    plt.xticks(np.arange(4), ['','younger', 'older',''])
    plt.title('Proportions of transmissions from younger v.s. older men')
        
    # make the whiskers more obvious
    vplot['cmins'].set_linewidths(3)
    vplot['cmins'].set_color('black')
    vplot['cmaxes'].set_linewidths(3)
    vplot['cmaxes'].set_color('black')
    vplot['cmeans'].set_linewidths(3)
    vplot['cmeans'].set_color('black')
    vplot['cbars'].set_linewidths(3)
    vplot['cbars'].set_color('black')
    
#    colors = ['pink', 'lightblue']
#    for patch, color in zip(bplot['boxes'], colors):
#        patch.set_facecolor(color)
        
    if savepath is not None:
        plt.savefig(savepath)  
    plt.show()
        
    
    return


#%%
    
poolWeights('trans_flow/', list(range(1,101)), savepath='weightsMF_less_youner_men.pdf')

poolWeights('trans_flow/', list(range(101,201)), savepath='weightsMF_more_youner_men.pdf')


#%%
# 01/12/2022:
# function to pool weights and save to csv file

# 01/24/2022:
# update to include posterior median as well

# 01/25/2022:
# update to get 95% credible intervals as well

from os.path import exists

## helper function: get pop size N and scenario version 
def N_ver(n):
    if n <= 200:
        N = 100
    elif n <= 400:
        N = 200
    elif n <= 600:
        N = 600
    else:
        N = 800
        
    if (n % 200 >= 1) and (n % 200 <= 100):
        ver = 1
    else:
        ver = 2
        
    return N, ver

def savePoolWeights(res_dir, nums, last=500, stat = 'mean', run = 'later', getCI = False):
    
    weight_means = np.zeros((len(nums), 2))
    N_ls = []
    v_ls = []
    
    if getCI:
        weight_lbs = np.zeros((len(nums), 2))
        weight_ubs = np.zeros((len(nums), 2))
    
    for i in range(len(nums)):
        n = nums[i]
        N, ver = N_ver(n)
        ## accmmodate previous N=400 runs
        if run != 'later':
            N = 400
        flabel = str(n)+'_'+str(N)+'_v'+str(ver) if run=='later' else str(n)
        fpath = res_dir + 'weightMF_'+flabel+'.pkl' 
        if exists(fpath):
            # if results are returned for this one
            W = pkl.load(file=open(fpath,'rb'))
            chain = np.array(W[-last:])[:,:2]
            # 01/24/2022 update: allow option to do median as well
            if stat == 'mean':
                w_means = np.mean(chain, axis=0)
            else:
                w_means = np.median(chain, axis=0)
            if ver == 1:
                # weight on younger men on left side
                w_means = w_means[::-1]
            # 01/25/2022 update: get 95% credible intervals
            if getCI:
                w_lb = np.quantile(chain, 0.025, axis=0)
                w_ub = np.quantile(chain, 0.975, axis=0)
                if ver == 1:
                    w_lb = w_lb[::-1]
                    w_ub = w_ub[::-1]
            
            
        else:
            # if no results are returned
            # fill with 0s...
            w_means = np.array((0.0,0.0))
            
            if getCI:
                w_lb = w_ub = np.array((0.0, 0.0))
            
        weight_means[i,:] = w_means
        
        if getCI:
            weight_lbs[i,:] = w_lb
            weight_ubs[i,:] = w_ub
        
        N_ls.append(N)
        v_ls.append(ver)
    
#    dat_dic = {'N': N_ls, 'scenario': v_ls, 
#               'younger_weight': weight_means[:,0], 
#               'older_weight': weight_means[:,1]}   
    
    # re-format data frame to match that in R processing
    if getCI:
        dat_dic = {'N': N_ls*2, 'scenario': v_ls*2, 
               'weight': ['younger' for n in nums] + ['older' for n in nums],
               'means': np.concatenate((weight_means[:,0],weight_means[:,1])),
               'ubs': np.concatenate((weight_ubs[:,0],weight_ubs[:,1])),
               'lbs': np.concatenate((weight_lbs[:,0],weight_lbs[:,1]))}
    else:
        dat_dic = {'N': N_ls*2, 'scenario': v_ls*2, 
                   'weight': ['younger' for n in nums] + ['older' for n in nums],
                   'means': np.concatenate((weight_means[:,0],weight_means[:,1]))}  
    
    return pd.DataFrame(dat_dic)
        
#%%
pool_weights =  savePoolWeights('trans_flow_v3/', list(range(1,801)))   

# save to csv for use in R
pool_weights.to_csv('pooled_weights.csv', index=False, index_label=False)

# 01/24/2022: extract posterior medians as well
median_weights = savePoolWeights('trans_flow_v3/', list(range(1,801)), stat = 'median')
median_weights.to_csv('pooled_median_weights.csv', index = False, index_label=False)  

## posterior medians for the previous N=400 runs as well
median_weights_400 = savePoolWeights('trans_flow/', list(range(1,201)), stat = 'median', run = 'prev')
median_weights_400.to_csv('pooled_median_weights_400.csv', index = False, index_label=False)

## 01/25/2022: update to get posterior CIs
## large run
pool_weights =  savePoolWeights('trans_flow_v3/', list(range(1,801)), getCI = True)
pool_weights.to_csv('pooled_weights_CIs.csv', index=False, index_label=False) 

## previous N=400 run
pool_weights_400 = savePoolWeights('trans_flow/', list(range(1,201)), run = 'prev', getCI = True)
pool_weights_400.to_csv('pooled_weights_CIs_400.csv', index=False, index_label=False)


#%%

# plot the proportions of transmission events
def poolCs(res_dir, nums, last=500, savepath=None):
    
    def tabulate(C):
        counts = np.empty(shape=2)
        for k in range(1,3):
            counts[k-1] = np.sum(C==k)/np.sum(C!=0)
        return counts
    
    C_means = np.zeros((len(nums), 2))
    
    for i in range(len(nums)):
        n = nums[i]
        fpath = res_dir + 'C_'+str(n)+'.pkl'
        Cs = pkl.load(file=open(fpath,'rb'))[-last:]
        
        all_counts = np.apply_along_axis(tabulate, 1, Cs)
        Counts_mean = np.mean(all_counts,axis=0)
        
        C_means[i,:] = Counts_mean
        
    data = [C_means[:,0],C_means[:,1]]
    plt.figure(figsize=(6,4))
        
#    bplot = plt.boxplot(data, vert = True, patch_artist=True,
#                labels=['younger','older'])
    
    vplot = plt.violinplot(data, showmeans=True, widths=0.5)
    
    plt.grid(axis='y')
    plt.xticks(np.arange(4), ['','M->F', 'F->M',''])
    plt.title('Proportions of identified transmission events')
    
    plt.ylim(0.3,0.7)
    
    # make the whiskers more obvious
    vplot['cmins'].set_linewidths(3)
    vplot['cmins'].set_color('black')
    vplot['cmaxes'].set_linewidths(3)
    vplot['cmaxes'].set_color('black')
    vplot['cmeans'].set_linewidths(3)
    vplot['cmeans'].set_color('black')
    vplot['cbars'].set_linewidths(3)
    vplot['cbars'].set_color('black')
    
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    
    return


#%%
poolCs('trans_flow/', list(range(1,101)), savepath='props_trans_events_equal.pdf')

poolCs('trans_flow/', list(range(101,201)), savepath='props_trans_events_moreMF.pdf')

#%%
# 01/12/2022: function to save pooled Cs

# 01/24/2022: allow extracting posterior medians 
# 01/25/2022: get posterior CIs as well
def savePoolCs(res_dir, nums, last=500, stat = 'mean', run = 'later', getCI = False):
    
    def tabulate(C):
        counts = np.empty(shape=2)
        for k in range(1,3):
            counts[k-1] = np.sum(C==k)/np.sum(C!=0)
        return counts
    
    C_means = np.zeros((len(nums), 2))
    N_ls = []
    v_ls = []
    
    if getCI:
        C_ubs = np.zeros((len(nums), 2))
        C_lbs = np.zeros((len(nums), 2))
    
    for i in range(len(nums)):
        n = nums[i]
        N, ver = N_ver(n)
        ## some massaging to accommodate the previous N=400 runs
        if run != 'later':
            N = 400
        flabel = str(n)+'_'+str(N)+'_v'+str(ver) if run=='later' else str(n)
        fpath = res_dir + 'C_'+flabel+'.pkl'
        
        if exists(fpath):
            # if results are returned
            try:
                Cs = pkl.load(file=open(fpath,'rb'))[-last:]
                all_counts = np.apply_along_axis(tabulate, 1, Cs)
                if stat == 'mean':
                    Counts_mean = np.mean(all_counts,axis=0)
                else:
                    Counts_mean = np.median(all_counts,axis=0)
                    
                if getCI:
                    Counts_lb = np.quantile(all_counts, 0.025, axis=0)
                    Counts_ub = np.quantile(all_counts, 0.975, axis=0)
                    
            except:
                print('Something went wrong when unpickling', fpath)
                # fill in zero for weird stuff as well
                Counts_mean = np.array((0.0, 0.0))
                
                if getCI:
                    Counts_ub = Counts_lb = np.array((0.0, 0.0))
            
        else:
            # if no results are returned
            # fill in with zeros
            Counts_mean = np.array((0.0, 0.0))
            
            if getCI:
                    Counts_ub = Counts_lb = np.array((0.0, 0.0))
        
        C_means[i,:] = Counts_mean
        N_ls.append(N)
        v_ls.append(ver)
        
        if getCI:
            C_ubs[i,:] = Counts_ub
            C_lbs[i,:] = Counts_lb
    
#    dat_dic = {'N': N_ls, 'scenario': v_ls, 
#               'MF_C': C_means[:,0], 
#               'FM_C': C_means[:,1]}   
#    
    # update data frame format to match what I already have for R processing
    dat_dic = {'N': N_ls*2, 'scenario': v_ls*2, 
               'weight': ['M->F' for n in nums] + ['F->M' for n in nums],
               'means': np.concatenate((C_means[:,0],C_means[:,1]))}  
    
    if getCI:
        dat_dic['ubs'] = np.concatenate((C_ubs[:,0],C_ubs[:,1]))
        dat_dic['lbs'] = np.concatenate((C_lbs[:,0],C_lbs[:,1]))
    
    return pd.DataFrame(dat_dic)

#%%
# 01/12/2022: pool and save
pool_Cs =  savePoolCs('trans_flow_v3/', list(range(1,801)))   

# save to csv for use in R
pool_Cs.to_csv('pooled_Cs.csv', index=False, index_label=False)

# 01/24/2022: get post medians as well
median_Cs = savePoolCs('trans_flow_v3/', list(range(1,801)), stat = 'median')   

# save to csv for use in R
median_Cs.to_csv('pooled_median_Cs.csv', index=False, index_label=False)

## also do this with previous N=400 runs
median_Cs_400 = savePoolCs('trans_flow/', list(range(1,201)), stat = 'median', run = 'prev') 
median_Cs_400.to_csv('pooled_median_Cs_400.csv', index=False, index_label=False)

# 01/25/2022: get posterior CIs along with means
## the later runs
pool_Cs_CIs = savePoolCs('trans_flow_v3/', list(range(1,801)), getCI = True)
pool_Cs_CIs.to_csv('pooled_Cs_CIs.csv', index=False, index_label=False)

## the N=400 runs
pool_Cs_CIs_400 = savePoolCs('trans_flow/', list(range(1,201)), run = 'prev', getCI = True)
pool_Cs_CIs_400.to_csv('pooled_Cs_CIs_400.csv', index=False, index_label=False)



#%%
def plotProps(Cs, savepath=None):
    
    def tabulate(C):
        counts = np.empty(shape=2)
        for k in range(1,3):
            counts[k-1] = np.sum(C==k)/np.sum(C!=0)
        return counts
                
    #Cs = np.array(self.chains['C'][s:])
    all_counts = np.apply_along_axis(tabulate, 1, Cs)
    Counts_mean = np.mean(all_counts,axis=0)
    Counts_std = np.std(all_counts,axis=0)
    Counts_CI = np.quantile(all_counts,[0.025,0.975], axis=0)
    
    ## subtract the mean to get the errorbar width
    Counts_bars = copy(Counts_CI)
    Counts_bars[0,:] = Counts_mean - Counts_bars[0,:]
    Counts_bars[1,:] = Counts_bars[1,:] - Counts_mean
    
    ind = np.arange(len(Counts_mean))
    plt.bar(ind, Counts_mean, 0.4, yerr = list(Counts_bars),
            error_kw=dict(lw=3, capsize=10, capthick=3), 
            color=["#F8766D", "#7CAE00"])
#    plt.errorbar(ind, Counts_mean, yerr = list(Counts_bars), 
#                 fmt='o', capthick=5)
    plt.title('Proportions of transmission events (w/ 95% CI)')
    plt.xticks(ind, ('MF', 'FM'))
    plt.ylim(0,0.65)
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    
    return Counts_CI


#  (2) plot weights of younger men vs older men
def plotWeights(W, top=2, oldInd=0, savepath=None):
    # default: only plot the top 2 components (one is younger, one is older)
    # W: model.chain['weightMF'], for example
    chain = np.array(W)[:,:top]
    w_means = np.mean(chain, axis=0)
    w_CI = np.quantile(chain, [0.025,0.975], axis=0)
    
    w_bars = copy(w_CI)
    w_bars[0,:] = w_means - w_bars[0,:]
    w_bars[1,:] = w_bars[1,:] - w_means
    
    if oldInd == 0:
        # put "younger" to the left
        w_bars = w_bars[:,::-1]
        w_means = w_means[::-1]
    
    ind = np.arange(len(w_means))
    plt.bar(ind, w_means, 0.4, yerr=list(w_bars),
            error_kw=dict(lw=3, capsize=10, capthick=3),
            color=["#7CAE00","#00BFC4"])
    plt.title('Proportions of transmissions from younger v.s. older men (w/ 95% CI)')
    plt.xticks(ind, ('younger','older'))
         
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()          
    
    return

   

