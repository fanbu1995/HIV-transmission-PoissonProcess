#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:17:49 2020

@author: fan
"""
#%%
import numpy as np
from numpy.linalg import inv
from numpy.random import choice
from scipy.special import logit, expit
from scipy.stats import multivariate_normal, norm, truncnorm
from scipy.stats import wishart#, invwishart
from scipy.stats import dirichlet
from sklearn.cluster import KMeans

#import pandas as pd
from copy import copy

# 10/11/2020 update
# numpy.random new generator...
from numpy.random import default_rng
rng = default_rng()

## 09/15/2021 UPDATE
## treat L=1 or D=1/0 cases specially
## do not include those into the L or D score models

#%%

# 1-d Gaussian stuff (score model)

## linked score

def initializeLinkedScore(L, initThres = 0.6, indsNoL = None):
    '''
    Initialize things based on thresholding on the linked score;
    Returns:
        - logit transformed link score
        - indices of pairs that are selected in the point processes
        - initial value of muL, gammaL (inverse of sigma^2_l)
    L: length N linked scores of all pairs
    indsNoL: indices of points excluced from L model
    '''
    all_inds = set(range(len(L)))
    inds = np.where(L > initThres)[0]
    
    if indsNoL is not None:
        all_inds = all_inds - set(indsNoL)
        inds = list(set(inds) - set(indsNoL))
    
    all_inds = list(all_inds)
    
    L[all_inds] = logit(L[all_inds])
    L[indsNoL] = 0 # fill in dummy 0's for those with L==1
    Thres = logit(initThres)
    
    Lsel = L[all_inds]
    muL = np.mean(Lsel[Lsel > Thres])
    
    deMean = np.where(Lsel > Thres, Lsel - muL, Lsel)
    
    gammaL = 1/np.mean(deMean ** 2)
    
    return L, inds, muL, gammaL


def updateLModel(L, indsMF, indsFM, muL, gammaL, gammaPrior, indsNoL = None):
    '''
    Update linked score model (muL and gammaL) given the point configurations
    Returns muL and gammaL
    L: length N linked scores (transformed) of all pairs
    indsMF: indices of points in the MF process
    indsFM: indices of points in the FM process
    gammaPrior: a dictionary of prior for gammaL, "nu0" and "sigma0"
    indsNoL: indices of points excluced from L model
    '''
    
    # 10/11/2020 update: only update when MF+FM is non-empty
    
    # 09/15/2021 update
    if indsNoL is not None:
        inds = list(set(list(indsMF) + list(indsFM)) - set(indsNoL))
    else:
        inds = list(indsMF) + list(indsFM)
    
    if len(inds) > 0:
        mu_mean = np.mean(L[inds])
        mu_std = 1/np.math.sqrt(len(inds) * gammaL)
    
        muL = truncnorm(a=(0-mu_mean)/mu_std, b=np.inf).rvs() * mu_std + mu_mean
    
    #deMean = L
    deMean = copy(L)
    deMean[inds] = deMean[inds] - muL
    
    if indsNoL is not None:
        all_inds = list(set(range(len(L))) - set(indsNoL))
        deMean = deMean[all_inds]
    
    SS = np.sum(deMean ** 2)
    
    gammaL = np.random.gamma((gammaPrior['nu0'] + len(L))/2, 
                             2/(gammaPrior['nu0'] * gammaPrior['sigma0'] + SS))
    
    return muL, gammaL

def evalLLikelihood(L, indsMF, indsFM, muL, gammaL, indsNoL = None, 
                    subset=None, log=True):
    '''
    Evaluate the linked score component of the likelihood (on a subset of entries);
    Returns length len(L) (or len(subset)) array of (log-)likelihood
    L: length N linked scores (transformed) of all pairs
    indsMF: indices of points in the MF process
    indsFM: indices of points in the FM process
    muL, gammaL: parameters of the L model
    indsNoL: indices of points excluded from the L score model (L=1 points) - for those return 0
    subset: list of SORTED indices (if None, then evaluate likelihood on all entries)
    log: bool, output log-likelihood?
    '''
    # get the indices in either point process
    #inds= list(E_MF.keys()) + list(E_FM.keys())
    inds = list(indsMF) + list(indsFM)
    
    if subset is not None:
        indsIn = list(set(subset) & set(inds))
        indsOut = list(set(subset) - set(inds))
        res = np.empty(len(subset))
        indices = np.array(subset)
    else:
        indices = np.arange(len(L))
        indsIn = inds
        indsOut = list(set(indices) - set(inds))
        res = np.empty(len(L))
        
    ## 09/15/2021 update
    if indsNoL is not None:
        indsNull = np.where(np.in1d(indices, indsNoL))[0]
        indsIn = list(set(indsIn) - set(indsNoL))
        indsOut = list(set(indsOut) - set(indsNoL))
        
    sd = 1/np.math.sqrt(gammaL)  
    #logDensIn = norm(loc=muL, scale=sd).logpdf(L[indsIn]) if len(indsIn) > 0 else 
    if len(indsIn) > 0:
        logDensIn = norm(loc=muL, scale=sd).logpdf(L[indsIn])
        res[np.searchsorted(indices, indsIn)] = logDensIn
    if len(indsOut) > 0:
        logDensOut = norm(loc=0, scale=sd).logpdf(L[indsOut])
        res[np.searchsorted(indices, indsOut)] = logDensOut
    ## then fill in those null entries (return 0 as loglik)
    if indsNoL is not None:
        res[indsNull] = 0
        
    if not log:
        res = np.exp(res)
        
    return res
        

## direction score
## 01/06/2021 change: add a Dthres argument (default 0.5, same as before)
##                    AND, do the subsetting BEFORE logit transform
    
def initializeDirectScore(D, inds, Dthres = 0.5, indsNoD = None):
    '''
    Initialize direction score stuff based on thresholding results of linked score;
    Returns:
        - logit transformed direction score
        - indices of pairs that are selected in each point process
        - initial value of muNegD, muD, gammaD (inverse of sigma^2_d)
    D: length N linked scores of all pairs (in same order as L)
    Dthres: the threshold for allocating points to MF and FM surfaces
    indsNoD: indices of points excluded from D model
    
    '''
    
    # get MF and FM indices first
    inds = set(inds)
    indsMF = inds & set(np.where(D > Dthres)[0])
    indsFM = inds - indsMF
    
    all_inds = set(range(len(D)))
    
    ## 09/15/2021
    if indsNoD is not None:
        inds = inds - set(indsNoD)
        all_inds = all_inds - set(indsNoD)
        indsMF = indsMF - set(indsNoD)
        indsFM = indsFM - set(indsNoD)
        
    indsMF = list(indsMF)
    indsFM = list(indsFM)
    inds = list(inds)
    all_inds = list(all_inds)
    
    # and then transform and get mixture stuff
    D[all_inds] = logit(D[all_inds])
    D[indsNoD] = 0 # fill in 0 values at those NULL spots
    
    muD = np.mean(D[indsMF])
    muNegD = np.mean(D[indsFM])
    
    Dsel = D[list(inds)]
    deMean = np.where(Dsel > 0, Dsel-muD, Dsel-muNegD)
    gammaD = 1/np.mean(deMean ** 2)
    
    return D, indsMF, indsFM, muD, muNegD, gammaD

## 09/15/2021:
# make the fixed D centers play an explicit role inside the udpate function

def updateDModel(D, indsMF, indsFM, muD, muNegD, gammaD, gammaPrior, 
                 fixmu=False, indsNoD = None):
    '''
    Update linked score model (muL and gammaL) given the point configurations
    Returns muD, muNegD, gammaD
    D: length N MF-direction scores (transformed) of all pairs
    indsMF: indices of points in the MF process
    indsFM: indices of points in the FM process
    gammaPrior: a dictionary of prior for gammaL, "nu0" and "sigma0"
    fixmu: True or False, if True, fix muD and muNegD without(!) any updating
    indsNoD: indices of points excluded from D model
    '''
    
    # 10/11/2020 update: only make updates on non-empty sets
    
    if fixmu:
        pass
    else:
        # update do the updating of muD and muNegD if the mu's are not fixed
        if indsNoD is not None:
            indsMF = list(set(indsMF) - set(indsNoD))
            indsFM = list(set(indsFM) - set(indsNoD))
        else:
            indsMF = list(indsMF) 
            indsFM = list(indsFM)
    
        if len(indsMF) > 0:
            muD_mean = np.mean(D[indsMF])
            muD_std = 1/np.math.sqrt(len(indsMF) * gammaD)
            muD = truncnorm(a=(0-muD_mean)/muD_std, b=np.inf).rvs() * muD_std + muD_mean
    
        if len(indsFM) > 0:
            muNegD_mean = np.mean(D[indsFM])
            muNegD_std = 1/np.math.sqrt(len(indsFM) * gammaD)
            muNegD = truncnorm(a=-np.inf, b=(0-muNegD_mean)/muNegD_std).rvs() * muNegD_std + muNegD_mean
     
    # update gammaD as before
    
    #deMean = D
    deMean = copy(D)
    deMean[indsMF] = deMean[indsMF] - muD
    deMean[indsFM] = deMean[indsFM] - muNegD
    
    if indsNoD is not None:
        all_inds = list(set(range(len(D))) - set(indsNoD))
        deMean = deMean[all_inds]
    
    SS = np.sum(deMean ** 2)
    
    gammaD = np.random.gamma((gammaPrior['nu0'] + len(D))/2, 
                             2/(gammaPrior['nu0'] * gammaPrior['sigma0'] + SS))
    
    return muD, muNegD, gammaD


def evalDLikelihood(D, indsMF, indsFM, muD, muNegD, gammaD, indsNoD = None,
                    subset=None, log=True):
    '''
    Evaluate the direction score component of the likelihood (on a subset of entries);
    Returns length len(D) (or len(subset)) array of (log-)likelihood
    D: length N direction scores (transformed) of all pairs
    indsMF: indices of points in the MF process
    indsFM: indices of points in the FM process
    muD, muNegD, gammaD: parameters of the D model
    indsNoD: indices of points excluded from the D score model (D = 1 or 0) - for those return 0
    subset: list of SORTED indices (if None, then evaluate likelihood on all entries)
    log: bool, output log-likelihood?
    '''
    # get the indices in each point process
    indsMF = list(indsMF) 
    indsFM = list(indsFM)
    
    # get indices in MF, MF and out
    if subset is not None:
        indsMF = list(set(subset) & set(indsMF))
        indsFM = list(set(subset) & set(indsFM))
        indsOut = list(set(subset) - (set(indsMF) | set(indsFM)))
        res = np.empty(len(subset))
        indices = np.array(subset)
    else:
        indices = np.arange(len(D))
        indsOut = list(set(indices) - (set(indsMF) | set(indsFM)))
        res = np.empty(len(D))
        
    ## 09/15/2021
    if indsNoD is not None:
        indsNull = np.where(np.in1d(indices, indsNoD))[0]
        indsMF = list(set(indsMF) - set(indsNoD))
        indsFM = list(set(indsFM) - set(indsNoD))
        indsOut = list(set(indsOut) - set(indsNoD))
        
    sd = 1/np.math.sqrt(gammaD)  

    if len(indsMF) > 0:
        logDensMF = norm(loc=muD, scale=sd).logpdf(D[indsMF])
        res[np.searchsorted(indices, indsMF)] = logDensMF
    if len(indsFM) > 0:
        logDensFM = norm(loc=muNegD, scale=sd).logpdf(D[indsFM])
        res[np.searchsorted(indices, indsFM)] = logDensFM
    if len(indsOut) > 0:
        logDensOut = norm(loc=0, scale=sd).logpdf(D[indsOut])
        res[np.searchsorted(indices, indsOut)] = logDensOut
    ## then fill in those null entries (return 0 as loglik)
    if indsNoD is not None:
        res[indsNull] = 0
        
    if not log:
        res = np.exp(res)
        
    return res

#%%
   
# test score model update
    
if __name__ == '__main__':
#    ## test initialization
#    L = (1-0.3)* np.random.random_sample(100) + 0.3
#    D = np.random.random_sample(100)
#    
#    Ltrans, inds, muL, gammaL = initializeLinkedScore(L, initThres = 0.6)
#    Dtrans, indsMF, indsFM, muD, muNegD, gammaD = initializeDirectScore(D, inds)
#        
#    ## test update
#    gaPrior = {'nu0': 2, 'sigma0': 1}
#    ## completely made up points...
#    E_MF = dict(zip(indsMF, np.random.random_sample(len(indsMF))))
#    E_FM = dict(zip(indsFM, np.random.random_sample(len(indsFM))))
#    
#    print(updateLModel(Ltrans, E_MF, E_FM, muL, gammaL, gaPrior))
#    print(updateDModel(Dtrans, E_MF, E_FM, muD, muNegD, gammaD, gaPrior))
    
    Ltrans, inds, muL, gammaL = initializeLinkedScore(L, initThres = 0.6)
    Dtrans, indsMF, indsFM, muD, muNegD, gammaD = initializeDirectScore(D, inds)
    
    print(Ltrans)
    print(Dtrans)
    
    E_MF = {i:v for i,v in E.items() if i in range(50)}
    E_FM = {i:v for i,v in E.items() if i in range(50,100)}
    
    gaPrior = {'nu0': 2, 'sigma0': 1}
    
    maxIter = 1000
    
    params = {'muL': [], 'gammaL':[], 'muD': [], 'muNegD': [], 'gammaD': []}
    
    for it in range(maxIter):
        muL, gammaL = updateLModel(Ltrans, E_MF, E_FM, muL, gammaL, gaPrior)
        params['muL'].append(muL); params['gammaL'].append(gammaL)
        
        muD, muNegD, gammaD = updateDModel(Dtrans, E_MF, E_FM, muD, muNegD, gammaD, gaPrior)
        params['muD'].append(muD); params['muNegD'].append(muNegD)
        params['gammaD'].append(gammaD)

    print(Ltrans)
    print(Dtrans)

#%%

# The point process stuff

def initializePP(E, indsMF, indsFM):
    '''
    Initialize MF and FM point process configurations.
    Returns:
        #- E_MF, E_FM, E_0: dictionary of (a_M, a_F) points on each type of surface
        - gamma: the scale for the entire process
        - probs: the length-3 vector of type probabilities/proportions
    E: dictionary of all (a_M, a_F) points (for all the pairs in data)
    indsMF, indsFM: some assignment of indices in MF and FM surfaces
    '''
    
#    E_MF = {pair: age for pair, age in E.items() if pair in indsMF}
#    E_FM = {pair: age for pair, age in E.items() if pair in indsFM}
#    E_0 = {pair: age for pair, age in E.items() if (pair not in indsFM and pair not in indsMF)}
    
    gamma = len(E)
    
    probs = [len(E) - len(indsMF) - len(indsFM) ,len(indsMF), len(indsFM)]
    probs = np.array(probs)/gamma
    
    return gamma, probs


def updateProbs(C, probPrior):
    '''
    C: length N, array like type indicator for all points (value in 0,1,2)
        0=outside, 1=MF, 2=FM
    probPrior: length 3, array like prior (for the Dirichlet prior)
    '''
    unique, counts = np.unique(C, return_counts=True)
    typeCounts = dict(zip(unique,counts))
    
    alpha = copy(probPrior)
    
    for k in typeCounts:
        alpha[k] += typeCounts[k]
        
    return dirichlet(alpha).rvs()[0]


def getPoints(E, subset=None):
    '''
    Return a (n,p) array of the points in event set E (or a subset)
    E: dictionary of indice, age pair
    subset: list of subset indices
    
    #(UPDATE: return None instead of raising error when E is empty)
    '''
    if not E:
        # if E is empty, raise an Error
        raise ValueError('The point event set is empty!')
        #X = None
    else:
        #p = X.shape[1]
        if subset:
            E_sub = {i: age for i,age in E.items() if i in subset}
            X = np.array(list(E_sub.values()))
            #n = len(subset)
        else:
            X = np.array(list(E.values()))
            #n = len(E)
        #X = X.reshape((n,p))

    return X

#%%
if __name__ == '__main__':
    E = {i: (np.random.random_sample(),np.random.random_sample()) for i in range(100)}
    X = getPoints(E)
    print(X.shape)
    
    inds1 = choice(range(100), size=38, replace=False)
    inds2 = choice(list(set(range(100)) - set(inds1)), size = 30, replace=False)
    
    E1, E2, gam1, gam2 = initializePP(E, inds1, inds2)
    
    E1, E2, chosen = proposePP(E, E1, E2, 10)



#%%
    
# update 10/11/2020: try out DP GMM
 

# sample new components from base measure
# here still use (mu,precision)!!
def sampleNewComp(Knew, muPrior, precisionPrior):
    '''
    Knew: number of new components to generate
    muPrior: dictionary of prior mean and precision --> covariance
    precisionPrior: dictionary of prior df and invScale AND Scale
    
    return: a list of NEW components
    '''
    comps = []
    
    if Knew == 0:
        return comps
    
    muCov = inv(muPrior['precision'])
    #precisionScale = inv(precisionPrior['invScale'])
    
    #muCov = muPrior['covariance']
    #covarianceScale = precisionPrior['Scale']
    
    
    for k in range(Knew):
        #mu = rng.multivariate_normal(muPrior['mean'], muPrior['covariance'])
        mu = rng.multivariate_normal(muPrior['mean'], muCov)
        precision = wishart(precisionPrior['df'], precisionPrior['Scale']).rvs()
        #covariance = invwishart(precisionPrior['df'], precisionPrior['Scale']).rvs()
        
        comps.append((mu, precision))
        #comps.append((mu, covariance))
    
    return comps   


# re-order components (and re-label labels) by sizes of components
def relabel(labels, components, Kmax=10):
    have_labels, counts = np.unique(labels,return_counts=True)
    label_order = np.argsort(counts)[::-1]
    
    new_labels = np.empty_like(labels)
    new_components = list()
    
    for k in range(len(have_labels)):
        # re-label
        new_labels[labels==label_order[k]] = k
        # move around components
        new_components.append(components[label_order[k]])
        
    # also include those components not present in population
    other_comps = [components[k] for k in range(Kmax) if k not in have_labels]
    new_components.extend(other_comps)
    
    return new_labels, new_components


# initialize DP GMM 
# get some components via K-means
# and append some new (surplus) components without associated datapoints
def initializeDPGMM(X, muPrior, precisionPrior, K=3, Kmax=10):
    '''
    Initialize a finite Gaussian mixture model via k-means;
    Returns components (mean and precision matrix) and component labels
    X: (n,p) array of data
    K: number of components to initialize with
    Kmax: max number of components for the truncated DP GMM
    
    returns: a list of Kmax components (center and co)
    '''
    kmeans = KMeans(n_clusters=K).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    components = list()
    for k in range(K):
        # still use precision matrix!
        components.append((centers[k,:], inv(np.cov(X[labels==k,:],rowvar=False))))
        # use covariance instead!
        #components.append((centers[k,:], np.cov(X[labels==k,:],rowvar=False)))
    
    # re-order components by counts
    labels, components = relabel(labels, components, Kmax=K)
    
    # then add more surplus components if necessary
    if Kmax > K:
        new_comps = sampleNewComp(Kmax-K, muPrior, precisionPrior)
        components.extend(new_comps)
        
    return components, labels


# update one Gaussian component
# inherited from before!
def updateOneComponent(X, mu, precision, muPrior, precisionPrior):
    '''
    X: (n,p) array of data
    mu: (p,1) array of current mean
    precision: (p,p) matrix of current precision
    muPrior: dictionary of prior mean and precision
    precisionPrior: dictionary of prior df and invScale
    '''
    
    n = X.shape[0]
    An_inv = inv(muPrior['precision'] + n * precision)
    Xsum = np.sum(X, axis=0)
    bn = muPrior['precision'].dot(muPrior['mean']) + precision.dot(Xsum)
    
    mu = multivariate_normal(An_inv.dot(bn), An_inv, allow_singular=True).rvs()
    
    S_mu = np.matmul((X-mu).T, X-mu)
    
    precision = wishart(precisionPrior['df'] + n, 
                        inv(precisionPrior['invScale'] + S_mu)).rvs()
    
    return mu, precision


# update all GMM components
#     i) update from data X if n_j > 0
#     ii) draw new components if n_j == 0
def updateGaussianComponents(X, Z, components, muPrior, precisionPrior):
    '''
    X: (n,p) array of data
    Z: length n, array like component indicator (only K distinct labels)
    components: list of (mu, precision) for Kmax Gaussian components
    muPrior: dictionary of prior mean and precision-->covariance
    precisionPrior: dictionary of prior df and invScale AND Scale
    
    Assume that
        - Z has K distinct values, 0,1,...,K-1
        - The labels are already ordered by component counts!
        - components has length Kmax (the last Kmax-K components have no data points)
    '''
    Kmax = len(components)
    
    have_labels = np.unique(Z)
    K = len(have_labels)
    
    for k in range(K):
        subX = X[Z==k,:]
        if subX.shape[0] > 0:
            mu, precision = components[k]
            components[k] = updateOneComponent(subX, mu, precision, 
                      muPrior, precisionPrior)
    
    if Kmax > K:
        components[K:Kmax] = sampleNewComp(Kmax-K, muPrior, precisionPrior)
            
    return components


# the prob vector function
def getProbVector(p):
    
    # some "anti-infinity" truncation to address numerical issues
    
    p[p==np.inf] = 3000
    p[p==-np.inf] = -3000
    
    p = np.exp(p - np.max(p))
    
    #print(p)
    
    return p/p.sum()


# update component indicators
# 10/11/2020: need to do this for each surface, then combine!
# also: still use (mu, precision)
def updateComponentIndicator(X, weight, components):
    '''
    X: (n,p) array of data
    weight: the mixture weight vector (for the surface that X lands on)
    components: list of (mu, precision) for K Gaussian components
    
    09/29 change: each component is (mu, covariance) instead
    
    (05/13 fix: use weights in indicator update! previous version was wrong)
    
    08/29 addtion: relabel the indicators and components by descending counts
    '''
    K = len(components)
    n = X.shape[0]
    
    logDens = np.empty((K,n))
    
    for k in range(K):
        mu, precision = components[k]
        MVN = multivariate_normal(mu, inv(precision), allow_singular=True)
        logDens[k,:] = MVN.logpdf(X) + np.log(weight[k])
#        logProb = MVN.logpdf(X)
#        if np.any(np.isnan(logProb)):
#            print(mu, precision)
#            raise ValueError("NaN in log likelihood!")
#        else:
#            logDens[k,:] = logProb
        
    Z = np.apply_along_axis(lambda v: choice(range(K), replace=False, 
                                             p=getProbVector(v)), 0, logDens)
    
    #print(Z)
    
    # relabel for later use!
    # (commented out because this needs to be done in the main fit function)
    #Z, components = relabel(Z, components, Kmax=len(components))
    
    return Z


# update component weights
# 10/11/20: do this on each surface, with labels Z "shared"
def updateMixtureWeight(Z, alpha, Kmax=10):
    '''
    Z: length n, array like component indicator
    alpha: the precision parameter for DP
    
    Assume that Z is labeled properly with descending counts
    (this may be violated in the Hierarchical 3-surface model)
    
    return: updated weight vector
    
    (Update following Chunlin Ji et al. 2009)
    '''
    
    # count component sizes
    counts = np.empty(shape=Kmax)
    for k in range(Kmax):
        counts[k] = np.sum(Z==k)
    
    # calculate the v's
    V = np.empty(shape=Kmax)
    for k in range(Kmax-1):
        alpha_k = 1+counts[k]
        beta_k = alpha + np.sum(counts[(k+1):])
        V[k] = rng.beta(alpha_k, beta_k)
    V[Kmax-1] = 1
    
    # calculate mixture probs
    W = np.empty_like(V)
    W[0] = V[0]
    V[0] = 1-V[0]
    W[1] = V[1] * V[0]
    for k in range(1, Kmax-1):
        V[k] = V[k-1] * (1-V[k])
        W[k+1] = V[k+1] * V[k]
        
    return W
    
   
# update precision (alpha) for DP
def updateAlpha(K, N, alpha, alphaPrior):
    '''
    K: num of unique components currently
    N: total number of data points
    alpha: current value of alpha (>0)
    alphaPrior: dictionary of Gamma prior
        - "a": rate
        - "b": shape (inverse of scale!)
    
    returns a new draw of alpha
    
    source: Escobar and West 1995
    '''
    a = alphaPrior['a']; b = alphaPrior['b']
    
    # auxiliary param "eta"
    aux = rng.beta(alpha+1, N)
    
    # odds
    odds = (a+K-1)/(N*(b-np.log(aux)))
    
    pi_aux = odds/(1+odds)
    if rng.binomial(1,pi_aux) == 1:
        alpha = rng.gamma(a+K, 1/(b-np.log(aux)))
    else:
        alpha = rng.gamma(a+K-1, 1/(b-np.log(aux)))
        
    return alpha



#%%

# Gaussian mixture stuff (spatal density model)

# 10/11/2020: commented out a bunch to extend to DP prior

def initializeGMM(X, K=2):
    '''
    Initialize a finite Gaussian mixture model via k-means;
    Returns components (mean and precision matrix) and component labels
    X: (n,p) array of data
    K: number of components
    '''
    kmeans = KMeans(n_clusters=K).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    components = list()
    for k in range(K):
        components.append((centers[k,:], np.cov(X[labels==k,:],rowvar=False)))
        
    return components, labels


#def updateOneComponent(X, mu, precision, muPrior, precisionPrior):
#    '''
#    X: (n,p) array of data
#    mu: (p,1) array of current mean
#    precision: (p,p) matrix of current precision
#    muPrior: dictionary of prior mean and precision
#    precisionPrior: dictionary of prior df and invScale
#    '''
#    
#    n = X.shape[0]
#    An_inv = inv(muPrior['precision'] + n * precision)
#    Xsum = np.sum(X, axis=0)
#    bn = muPrior['precision'].dot(muPrior['mean']) + precision.dot(Xsum)
#    
#    mu = multivariate_normal(An_inv.dot(bn), An_inv).rvs()
#    
#    S_mu = np.matmul((X-mu).T, X-mu)
#    
#    precision = wishart(precisionPrior['df'] + n, 
#                        inv(precisionPrior['invScale'] + S_mu)).rvs()
#    
#    return mu, precision

#def updateGaussianComponents(X, Z, components, muPrior, precisionPrior):
#    '''
#    X: (n,p) array of data
#    Z: length n, array like component indicator
#    components: list of (mu, precision) for K Gaussian components
#    muPrior: dictionary of prior mean and precision
#    precisionPrior: dictionary of prior df and invScale
#    '''
#    K = len(components)
#    
#    for k in range(K):
#        subX = X[Z==k,:]
#        if subX.shape[0] > 0:
#            mu, precision = components[k]
#            components[k] = updateOneComponent(subX, mu, precision, 
#                      muPrior, precisionPrior)
#            
#    return components

#def getProbVector(p):
#    
#    # some "anti-infinity" truncation to address numerical issues
#    
#    p[p==np.inf] = 3000
#    p[p==-np.inf] = -3000
#    
#    p = np.exp(p - np.max(p))
#    
#    #print(p)
#    
#    return p/p.sum()

#def updateComponentIndicator(X, weight, components):
#    '''
#    X: (n,p) array of data
#    components: list of (mu, precision) for K Gaussian components
#    (05/13 fix: use weights in indicator update! previous version was wrong)
#    '''
#    K = len(components)
#    n = X.shape[0]
#    
#    logDens = np.empty((K,n))
#    
#    for k in range(K):
#        mu, precision = components[k]
#        MVN = multivariate_normal(mu, inv(precision))
#        logDens[k,:] = MVN.logpdf(X) + np.log(weight[k])
##        logProb = MVN.logpdf(X)
##        if np.any(np.isnan(logProb)):
##            print(mu, precision)
##            raise ValueError("NaN in log likelihood!")
##        else:
##            logDens[k,:] = logProb
#        
#    Z = np.apply_along_axis(lambda v: choice(range(K), replace=False, 
#                                             p=getProbVector(v)), 0, logDens)
#    return Z

#def updateMixtureWeight(Z, weightPrior):
#    '''
#    Z: length n, array like component indicator
#    weightPrior: length K, array like prior (for the Dirichlet prior)
#    '''
#    unique, counts = np.unique(Z, return_counts=True)
#    mixtureCounts = dict(zip(unique,counts))
#    
#    alpha = copy(weightPrior)
#    
#    for k in mixtureCounts:
#        alpha[k] += mixtureCounts[k]
#        
#    return dirichlet(alpha).rvs()[0]

def evalDensity(X, weight, components, log=True):
    '''
    Evaluate the entire density function (after mixture) on points X;
    Returns a length-n array of density/log-density
    X: (n,p) array of data
    weight: length K vector of mixture weights
    components: list of (mu, precision) for K Gaussian components
    '''
    
    n = X.shape[0]
    K = len(weight)
    
    mix_dens = np.empty((n,K))
    
    for k in range(K):
        mu, precision = components[k]
        MVN = multivariate_normal(mu, inv(precision), allow_singular=True)
        mix_dens[:,k] = MVN.pdf(X)
        
    #print(mix_dens)
        
    total_dens = np.sum(weight * mix_dens, axis=1)
    
    if log:
        total_dens = np.log(total_dens)
        
    #print(total_dens)
        
    return total_dens
#%% test
#x_test = np.random.randn(50,2) + 2

# initialize function
#components, Z = initializeGMM(x_test)

# update component function
#muP = {'mean': np.array([0,0]), 'precision': np.eye(2)}
#preP = {'df': 2, 'invScale': np.eye(2)*.01}
#
#updateOneComponent(x_test, np.array([0.1,0.1]), np.eye(2), muP, preP)

# update indicator function
#components = [(np.zeros(2), np.eye(2)), (np.ones(2), np.eye(2) * 0.01)]
#Z = updateComponentIndicator(x_test, components)
#Z.shape[0] == x_test.shape[0]

# update mixture weight function
#updateMixtureWeight(Z, np.ones(2))


#%% test out the whole process
if __name__ == "__main__":

    from time import perf_counter
    
    muP = {'mean': np.array([0,0]), 'precision': np.eye(2)}
    preP = {'df': 2, 'invScale': np.eye(2)*.0001}
    weightP = np.ones(2)
    
    x_1 = np.random.randn(100,2) + 10
    x_2 = np.random.randn(100,2) -10
    x_test = np.concatenate((x_1,x_2),axis=0)
    
    tic = perf_counter()
    
    components, Z = initializeGMM(x_test)
    
    maxIter = 100
    
    for i in range(maxIter):
        components = updateGaussianComponents(x_test, Z, components, 
                                              muP, preP)
        Z = updateComponentIndicator(x_test, components)
        w = updateMixtureWeight(Z, weightP)
        
    #log_dens = evalDensity(x_test[:10,:], w, components)
    #print("log likelihood of first 10 points: {:.4f}".format(log_dens))
    
    #print(evalDensity(x_test, w, components))
        
    elapsed = perf_counter() - tic
    
    print("Total time {:.4f} seconds, with {:.4f} seconds per iteration.".format(elapsed,elapsed/maxIter))
        
    # It seems to work...
    # But occassionally would encounter NaN in the log density??
    # Probably fixed...
    
#%%
# re-rest the Gaussian mixture model
#from time import perf_counter
#    
#muP = {'mean': np.array([0,0]), 'precision': np.eye(2)*.0001}
#preP = {'df': 2, 'invScale': np.eye(2)}
#weightP = np.ones(3)
#
#x_test = X[:100,:]
#
#tic = perf_counter()
#
#components, Z = initializeGMM(x_test,K=3)
#
#maxIter = 2000
#
#for i in range(maxIter):
#    components = updateGaussianComponents(x_test, Z, components, 
#                                          muP, preP)
#    Z = updateComponentIndicator(x_test, components)
#    w = updateMixtureWeight(Z, weightP)
#    
##log_dens = evalDensity(x_test[:10,:], w, components)
##print("log likelihood of first 10 points: {:.4f}".format(log_dens))
#
##print(evalDensity(x_test, w, components))
#    
#elapsed = perf_counter() - tic
#
#print("Total time {:.4f} seconds, with {:.4f} seconds per iteration.".format(elapsed,elapsed/maxIter))

#%%
# functions to simulate data
def simulateGMM(N, weight, components):
    comp_counts = np.random.multinomial(N, weight)
    data = None
    for k in range(len(weight)):
        if comp_counts[k] > 0:
            data_k = np.random.multivariate_normal(components[k][0], components[k][1], comp_counts[k])
            if data is None:
                data = data_k
            else:
                data = np.vstack((data,data_k))
    return data

def simulateLatentPoissonHGMM(N, Settings):
    '''
    Simulate a dataset with N pairs
    Return: E, L, D
    Settings: a giant dictionary with settings and parameters
        - 'N_MF', 'N_FM': number of points in each point process
        - 'muD', 'muNegD', 'muL': the score model means
        - 'gammaD', 'gammaL': the score model precisions (inverse variance)
        - 'components': length K list of GMM components (mean vector, precision matrix)
        - 'weightMF', 'weightFM', 'weight0': mixture weight of GMM on each process
    '''
    
    N_MF = Settings['N_MF']
    N_FM = Settings['N_FM']
    N_out = N - N_FM - N_MF
    
    assert N_MF + N_FM <= N
    
    # 1. Generate L and D
    Lin = norm(loc=Settings['muL'], scale = 1/np.sqrt(Settings['gammaL'])).rvs(N_MF + N_FM)
    Lout = norm(loc=0, scale = 1/np.sqrt(Settings['gammaL'])).rvs(N_out)
    L = expit(np.concatenate((Lin, Lout)))
    
    D_MF = norm(loc=Settings['muD'], scale = 1/np.sqrt(Settings['gammaD'])).rvs(N_MF)
    D_FM = norm(loc=Settings['muNegD'], scale = 1/np.sqrt(Settings['gammaD'])).rvs(N_FM)
    D_out = norm(loc=0, scale = 1/np.sqrt(Settings['gammaD'])).rvs(N_out)
    D = expit(np.concatenate((D_MF,D_FM,D_out)))
    
    # 2. Generate E
    ## Those who are in MF
    MFvalues = simulateGMM(N_MF, Settings['weightMF'], Settings['components'])
    Evalues = list(MFvalues)
    
    ## Those who are in FM
    FMvalues = simulateGMM(N_FM, Settings['weightFM'], Settings['components'])
    #FMvalues = FMvalues[:,::-1] # flip the age, so that it's always (a_M, a_F)
    Evalues.extend(list(FMvalues))

    ## Those who are outside
    ### randomly sample within the range of points already sampled
#    Mins = np.min(Evalues, axis=0)
#    Maxs = np.max(Evalues, axis=0)
#    AgeM = np.random.random_sample((N_out,1)) * (Maxs[0] - Mins[0]) + Mins[0]
#    AgeF = np.random.random_sample((N_out,1)) * (Maxs[1] - Mins[1]) + Mins[1]
#    Evalues.extend(list(np.hstack((AgeM,AgeF))))
    OutValues = simulateGMM(N_out, Settings['weight0'], Settings['components'])
    Evalues.extend(list(OutValues))

    
    ## put together
    E = dict(zip(range(N),Evalues))
    
    return E, L, D
    