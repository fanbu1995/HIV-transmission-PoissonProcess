#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 23:15:38 2021

@author: fan
"""

#%%

# 04/23/2021: modified to run 100-round simulations on server

# 01/11/2022: run more experiments with diff. sample sizes
# ---- do more data improve inference precision??



#%%

# 04/05/2021: run simulation experiments of the DPGMM model (3 surface one)
import os
#os.chdir('/Users/fan/Documents/Research_and_References/HIV_transmission_flow/')

# 10/11/2020: changed working directory
#os.chdir('/Users/fan/Documents/Research_and_References/HIV_transmission_flow/HIV_transmission_flow/')

from copy import copy#, deepcopy

import matplotlib.pyplot as plt

#%%
# the updated utils function for real data analysis
#from utilsHupdate import *

# import from the previous version of utils functions for simulations...
from utilsHupdatePrev import *


## 01/09/2021: a version with 3 separate surfaces (separate components and weights)


# just make sure the getProbVector function works properly!
def getProbVector(p):
    
    # some "anti-infinity" truncation to address numerical issues
    
    p[p==np.inf] = 3000
    p[p==-np.inf] = -3000
    
    p = np.exp(p - np.max(p))
    
    #print(p)
    
    return p/p.sum()


class LatentPoissonDPHGMM:
    def __init__(self, Priors, K=3, Kmax = 10, linkThreshold=0.6):
        '''
        Inialize an instance of the LatentPoisson Hierarchical GMM model;
        Priors: a big dictionary of all the priors
            - "gammaPP": prior dictionary for the NHPP scale (gamma); 
                need "n0" and "b0"
            - "probs": prior vector (length 3) for the surface probability/proportion vector
            - "muGMM": prior dictionary for the means in Gaussian Mixture; 
                need "mean" and "precision"
            - "precisionGMM": prior dictionary for the precision matrices in Gaussian Mixture;
                need "df" and "invScale"
            - "weight": prior vector (length K) for Gaussian Mixture weight;
            - "gammaScore": prior dictionary for inverse variance of the score models;
                need "nu0" and "sigma0"
            - "alpha": prior for the DP precision; need "a" and "b"
        '''
        self.name = "Latent Poisson Process with Dirichlet Process Gaussian Mixture density"
        self.linkInitialThreshold = linkThreshold
        # number of mixture components
        self.K = K
        self.Kmax = Kmax
        # prior part
        self.ScoreGammaPrior = Priors["gammaScore"]
        self.muPrior = Priors["muGMM"]
        self.precisionPrior = Priors["precisionGMM"]
        #self.weightPrior = Priors["weight"]
        self.PPGammaPrior = Priors["gammaPP"]
        self.probPrior = Priors["probs"]
        self.alphaPrior = Priors['alpha']
        # data part
        self.E = None # all the (a_M,a_F) pairs
        self.L = None # all the linked scores
        self.D = None # all the direction scores
        self.indsMF = None # indices on the MF surface
        self.indsFM = None # indices on the FM surface
        self.inds0 = None # indices for the outsider points
        #self.E_MF = None # event set on MF surface
        #self.E_FM = None # event set on FM surface
        #self.E_0 = None # event set on the outside
        # parameters
        self.muL = None
        self.muD = None
        self.muNegD = None
        self.gammaL = None
        self.gammaD = None
        self.gamma = None # the scale parameter for the entire NHPP
        self.probs = None # the surface probability/proportions vector
        self.C = None # the surface allocation vector for all events
        self.componentsMF = None # components for MF
        self.componentsFM = None # components for FM
        self.components0 = None # components for outside surface
        self.weightMF = None
        self.weightFM = None
        self.weight0 = None # GMM weights for the "outside" surface
        self.Z_MF = None # component indicator for MF process
        self.Z_FM = None # component indicator for MF process
        self.Z_0 = None # component indicator for the outside process
        self.alpha_MF = None # DP precision for the MF mixture components
        self.alpha_FM = None # DP precision for the FM mixture components
        self.alpha_0 = None # DP precision for the outside mixture components
        self.params_to_record = ['muL','muD', 'muNegD', 'gammaL', 'gammaD', 
                                 'N_MF', 'N_FM', 'gamma', 'probs', 'C', 
                                 'componentsMF', 'componentsFM', 'components0', 
                                 'weightMF', 'weightFM', 'weight0',
                                 'alpha_MF', 'alpha_FM', 'alpha_0']
        # log-likelihood
        #self.log-lik-terms = None # each pair's contribution to the log-likelihood
        self.log_lik = None # total log-likelihood
        # posterior inference (summary statistics and chains)
        self.maxIter = None
        self.burn = 0
        self.thin = 1
        self.chains = {param: list() for param in self.params_to_record}
            # a dictionary for parameter samples
        # 10/26/2020: also record the the likelihood in each iteration 
        # (in order to extract MAP estimate)
        self.chains['loglik'] = list()
            
    def evalLikelihood(self, subset=None):
        '''
        Evaluate likelihood
        Returns total log likelihood 
        (Currently no implementation on subset!!)
        '''

        # Right now: STUPID WAY - sum over individual entries
        # later might change
        
        LLik = np.sum(evalLLikelihood(self.L, self.indsMF, self.indsFM, self.muL, 
                                      self.gammaL, subset=subset, log=True))
        DLik = np.sum(evalDLikelihood(self.D, self.indsMF, self.indsFM, 
                                      self.muD, self.muNegD, 
                                      self.gammaD, subset=subset, log=True))
        
        X = getPoints(self.E)
        N = len(self.E)
        
        MFLik = np.sum(evalDensity(X[self.indsMF,:], self.weightMF, self.componentsMF, log=True)) if len(self.indsMF) > 0 else 0
        FMLik = np.sum(evalDensity(X[self.indsFM,:], self.weightFM, self.componentsFM, log=True)) if len(self.indsFM) > 0 else 0
        Lik0 =  np.sum(evalDensity(X[self.inds0,:], self.weight0, self.components0, log=True)) if len(self.inds0) > 0 else 0 
        
        counts = np.array([len(self.inds0), len(self.indsMF), len(self.indsFM)])
        
        
        total = LLik + DLik + MFLik + FMLik + Lik0 + counts.dot(np.log(self.probs))
        total += N * np.log(self.gamma) - np.log(range(1,N+1)).sum() - self.gamma
        
#        if subset is None:
#            to_add = (N_MF * np.log(self.gammaMF) + N_FM * np.log(self.gammaFM) - 
#                      np.log(range(1,N_MF+1)).sum() - np.log(range(1,N_FM+1)).sum())
#            total += to_add - (self.gammaMF + self.gammaFM)
#            self.log_lik = total
            
        return total
    
    def updateTypeIndicator(self):
        '''
        Update the type indicator "C" for each point in the dataset
        Returns a length-N vector of indicators (values in 0, 1, 2)
        '''

        N = len(self.E)
        indsall = list(range(N))
        
        condProbs = np.empty((N,3))
        
        # h=0 (all outside)
        condProbs[:,0] = (evalLLikelihood(self.L, [], [], self.muL, self.gammaL) + 
                 evalDLikelihood(self.D, [], [], self.muD, self.muNegD, self.gammaD) + 
                 evalDensity(getPoints(self.E), self.weight0, self.components0) +
                 np.log(self.probs[0]))
        
        # h=1 (all in MF)
        condProbs[:,1] = (evalLLikelihood(self.L, indsall, [], self.muL, self.gammaL) + 
                 evalDLikelihood(self.D, indsall, [], self.muD, self.muNegD, self.gammaD) + 
                 evalDensity(getPoints(self.E), self.weightMF, self.componentsMF) +
                 np.log(self.probs[1]))
        
        # h=2 (all in FM)
        condProbs[:,2] = (evalLLikelihood(self.L, [], indsall, self.muL, self.gammaL) + 
                 evalDLikelihood(self.D, [], indsall, self.muD, self.muNegD, self.gammaD) + 
                 evalDensity(getPoints(self.E), self.weightFM, self.componentsFM) +
                 np.log(self.probs[2]))
        
        self.C = np.apply_along_axis(lambda v: choice(range(3), replace=False, 
                                                      p=getProbVector(v)), 1, condProbs)

        
        return
        

    
    def fit(self, E, L, D, samples = 1000, burn = 0, thin = 1, random_seed = 42, 
            verbose = True, debugHack = False, 
            fixed_alloc = False, thresholds = [0.6,0.5],
            D_centers = [0.5, -0.5]):
        '''
        Fit the model via MCMC
        fixed_alloc: whether or not using fixed allocation with thresholds
        thresholds: [threshold for L score, threshold for D score (> ... --> MF)]
        D_centers: [muD, muNegD] - fixed mixture centers for easier stability
        '''
        # set up
        self.E = E
        self.L = L
        self.D = D
        N = len(E)
        #self.log-lik-terms = np.empty(len(E))
        self.burn = burn
        self.thin = thin
        self.maxIter = samples * thin + burn
        
        np.random.seed(random_seed)
        
        # (Take care of all the gamma draws at the beginning???)
        
        
        # initialize
        # 1) scores
        ## 01/06/2021 change: add "fixed allocation" option
        if fixed_alloc:
            thresL, thresD = thresholds
            self.L, inds, self.muL, self.gammaL = initializeLinkedScore(self.L, thresL)
            self.D, self.indsMF, self.indsFM, self.muD, self.muNegD, self.gammaD = initializeDirectScore(self.D, inds, thresD)
        else:
            self.L, inds, self.muL, self.gammaL = initializeLinkedScore(self.L, self.linkInitialThreshold)
            self.D, self.indsMF, self.indsFM, self.muD, self.muNegD, self.gammaD = initializeDirectScore(self.D, inds)
        # 2) the PP
        self.gamma, self.probs = initializePP(self.E, self.indsMF, self.indsFM)
        self.C = np.zeros(N)
        self.C[self.indsMF] = 1
        self.C[self.indsFM] = 2
        # 3) DP Gaussian components and weights
        
        ## 01/06/2021 change: 3 surfaces are treated separately (nothing is shared any more)
        ## get points 
        X = getPoints(self.E)
        
        ## some initial estimates of the alphas
        self.alpha_MF, self.alpha_FM, self.alpha_0 = rng.gamma(self.alphaPrior['a'], 1/self.alphaPrior['b'], size=3)
        
        # 3.1) MF surface
        X_MF = X[self.indsMF,:]
        ## components and indicators
        self.componentsMF, self.Z_MF = initializeDPGMM(X_MF, self.muPrior, 
                                                       self.precisionPrior, self.K, self.Kmax)
        ## GMM weights
        self.weightMF = updateMixtureWeight(self.Z_MF, self.alpha_MF, self.Kmax)
        
        ## update alpha
        K_MF = len(np.unique(self.Z_MF))
        N_MF = len(self.indsMF)
        self.alpha_MF = updateAlpha(K_MF, N_MF, self.alpha_MF, self.alphaPrior)
        
        # 3.2) FM surface
        X_FM = X[self.indsFM,:]
        ## components and indicators
        self.componentsFM, self.Z_FM = initializeDPGMM(X_FM, self.muPrior, 
                                                       self.precisionPrior, self.K, self.Kmax)
        ## GMM weights
        self.weightFM = updateMixtureWeight(self.Z_FM, self.alpha_FM, self.Kmax)
        
        ## update alpha
        K_FM = len(np.unique(self.Z_FM))
        N_FM = len(self.indsFM)
        self.alpha_FM = updateAlpha(K_FM, N_FM, self.alpha_FM, self.alphaPrior)
        
        # 3.1) outside surface
        self.inds0 = np.where(self.C == 0)[0]
        X_0 = X[self.inds0,:]
        ## components and indicators
        self.components0, self.Z_0 = initializeDPGMM(X_0, self.muPrior, 
                                                     self.precisionPrior, self.K, self.Kmax)
        ## GMM weights
        self.weight0 = updateMixtureWeight(self.Z_0, self.alpha_0, self.Kmax)
        
        ## update alpha
        K_0 = len(np.unique(self.Z_0))
        N_0 = len(self.inds0)
        self.alpha_FM = updateAlpha(K_0, N_0, self.alpha_0, self.alphaPrior)
        
        
        if(verbose):
            print('Initialization done!')
        
        # MCMC
        # 05/09 debug: hack it to fix everything else except E_MF, E_FM and see how it goes...
        for it in range(self.maxIter):
            ## 1. the score models
            # HACK it for debugging purposes:
            if debugHack:
                self.muL, self.gammaL = Settings['muL'], Settings['gammaL']
                self.muD, self.muNegD, self.gammaD = Settings['muD'], Settings['muNegD'], Settings['gammaD']
            else:
                ## 01/09/2021: fixed_alloc or fixed centers
                self.muL, self.gammaL = updateLModel(self.L, self.indsMF, self.indsFM, self.muL, 
                                                     self.gammaL, self.ScoreGammaPrior)
                
                self.muD, self.muNegD, self.gammaD = updateDModel(self.D, self.indsMF, self.indsFM, 
                                                                  self.muD, self.muNegD, 
                                                                  self.gammaD, self.ScoreGammaPrior)  
                
            # 01/09/2021: deal with stuff separately based on settings
                
            if fixed_alloc:
                # the inds and C are all fixed throughout, no need to update
                pass
            else:
                # fix the two centers for the D scores
                #self.muD, self.muNegD = D_centers
                
                # 04/05/2021: be honest and not fix anything...
                pass
                
                # 01/09/2021: try fixing muL too...
                # self.muL = 2.0
                    
                # 10/26/2020 try stuff
                # 1) fix the value of muD and muNegD to salvage things
                #self.muD = 0.5 # approximate thres = 0.622
                #self.muNegD = -0.5
                
                # 2) truncate muD and muNegD at a non-zero value
                # to stablize things
                #delta = 0.1
                #self.muD = self.muD if self.muD > delta else delta
                #self.muNegD = self.muNegD if self.muNegD < -delta else -delta
            
                # also make sure they are not too positive or negative
                #delta2 = 1.0
                #self.muD = self.muD if self.muD <= delta2 else delta2
                #self.muNegD = self.muNegD if self.muNegD >= -delta2 else -delta2
                    
                    
                ## 2. the point configurations
        
                ## 2.1 update event type allocation
                self.updateTypeIndicator()
        
                ## 2.2 update probs
                self.probs = updateProbs(self.C, self.probPrior)
        
                ## 2.3 bookkeeping
                self.indsMF = np.where(self.C == 1)[0]
                self.indsFM = np.where(self.C == 2)[0]
                self.inds0 = np.where(self.C == 0)[0]
        
                #self.E_MF = {pair: age for pair, age in self.E.items() if pair in self.indsMF}
                #self.E_FM = {pair: age for pair, age in self.E.items() if pair in self.indsFM}
                #self.E_0 = {pair: age for pair, age in self.E.items() if pair in inds0}
            
        
            ## 3. Update gamma
            self.gamma = np.random.gamma(self.PPGammaPrior['n0']+N, 1/(self.PPGammaPrior['b0']+1))
            
            ## 4. Update the DP Gaussian Mixture Model for the densities
            ## 01/09/2021 change: 3 surfaces separately!
            ### 4.1 MF surface
            if len(self.indsMF) > 0:
                # get surface points
                X_MF = X[self.indsMF,:]
                # update component indicators
                self.Z_MF = updateComponentIndicator(X_MF, self.weightMF, self.componentsMF)
                # re-order labels!
                self.Z_MF, self.componentsMF = relabel(self.Z_MF, self.componentsMF, self.Kmax)
                # update components
                self.componentsMF = updateGaussianComponents(X_MF, self.Z_MF, self.componentsMF, 
                                                             self.muPrior, self.precisionPrior)
                # update weights
                self.weightMF = updateMixtureWeight(self.Z_MF, self.alpha_MF, self.Kmax)
                # update alpha
                K_MF = len(np.unique(self.Z_MF))
                N_MF = len(self.indsMF)
                self.alpha_MF = updateAlpha(K_MF, N_MF, self.alpha_MF, self.alphaPrior)
                
            ### 4.2 the FM surface
            if len(self.indsFM) > 0:
                X_FM = X[self.indsFM,:]
                # update component indicators
                self.Z_FM = updateComponentIndicator(X_FM, self.weightFM, self.componentsFM)
                # re-order labels!
                self.Z_FM, self.componentsFM = relabel(self.Z_FM, self.componentsFM, self.Kmax)
                # update components
                self.componentsFM = updateGaussianComponents(X_FM, self.Z_FM, self.componentsFM, 
                                                             self.muPrior, self.precisionPrior)
                # update weights
                self.weightFM = updateMixtureWeight(self.Z_FM, self.alpha_FM, self.Kmax)
                # update alpha
                K_FM = len(np.unique(self.Z_FM))
                N_FM = len(self.indsFM)
                self.alpha_FM = updateAlpha(K_FM, N_FM, self.alpha_FM, self.alphaPrior)
            
            ### 4.3 the outsiders
            if len(self.inds0) > 0:
                X_0 = X[self.inds0,:]
                # update component indicators
                self.Z_0 = updateComponentIndicator(X_0, self.weight0, self.components0)
                # re-order labels!
                self.Z_0, self.components0 = relabel(self.Z_0, self.components0, self.Kmax)
                # update components
                self.components0 = updateGaussianComponents(X_0, self.Z_0, self.components0, 
                                                            self.muPrior, self.precisionPrior)
                # update weights
                self.weight0 = updateMixtureWeight(self.Z_0, self.alpha_0, self.Kmax)
                # update alpha
                K_0 = len(np.unique(self.Z_0))
                N_0 = len(self.inds0)
                self.alpha_0 = updateAlpha(K_0, N_0, self.alpha_0, self.alphaPrior)
            
            
            ## 5. Save parameter in chains if...
            if (it >= burn) & ((it+1-burn) % thin == 0):
                self.chains['muL'].append(self.muL)
                self.chains['muD'].append(self.muD)
                self.chains['muNegD'].append(self.muNegD)
                self.chains['gammaL'].append(self.gammaL)
                self.chains['gammaD'].append(self.gammaD)
                self.chains['N_MF'].append(len(self.indsMF))
                self.chains['N_FM'].append(len(self.indsFM))
                self.chains['gamma'].append(self.gamma)
                self.chains['probs'].append(self.probs)
                self.chains['C'].append(self.C)
                self.chains['componentsMF'].append(self.componentsMF)
                self.chains['componentsFM'].append(self.componentsFM)
                self.chains['components0'].append(self.components0)
                self.chains['weightMF'].append(self.weightMF)
                self.chains['weightFM'].append(self.weightFM)
                self.chains['weight0'].append(self.weight0)
                self.chains['alpha_MF'].append(self.alpha_MF)
                self.chains['alpha_FM'].append(self.alpha_FM)
                self.chains['alpha_0'].append(self.alpha_0)
                
                # 10/26/20: add loglik
                log_lik = self.evalLikelihood()
                self.chains['loglik'].append(log_lik)
                
                if verbose:
                    print('Parameters saved at iteration {}/{}. Log-likelihood={}'.format(it, self.maxIter, log_lik))
            
        return
    
    def plotChains(self, param, s=None, savepath=None):
        
        # updated 10/11/2020: add more plotting functions 
        # (inherited from the DPGMM version)
        
        
        if param.startswith('compo'):
            
            # 10/11/2020: adapted to the 3-surface case
            
            # a helper function for plotting
            def plotSurface(data, name, weights, components, savepath=None):
                # get the min, max range
                Amin = np.min(data); Amax = np.max(data)
                
                # specify range
                Amin = min(15,np.min(data)); Amax = max(50,np.max(data))
            
                # make density contour plot
                #Amin = 15.0; Amax = 50.0
                x = np.linspace(Amin, Amax)
                y = np.linspace(Amin, Amax)
                X, Y = np.meshgrid(x, y)
                XX = np.array([X.ravel(), Y.ravel()]).T
            
                Z = evalDensity(XX, weights, components, log=True)
                Z = Z.reshape(X.shape)
            
                plt.contourf(X,Y,Z)
            
                # overlay with the predicted age-pair points
                plt.scatter(data[:,0], data[:,1], c="black")
            
                plt.title('predicted log-density of the {} surface'.format(name))
                # 10/26/2020 fix: didn't reverse the order of age pair
                # so plot male age and female age instead
                #plt.xlabel('transmitter age')
                #plt.ylabel('recipient age')
                plt.xlabel('male age')
                plt.ylabel('female age')
                if savepath is not None:
                    plt.savefig(savepath)
                plt.show()
                return
            
            # plot each surface one by one
            chain = self.chains[param]
            if s >= len(chain) or s is None:
                s = -1
            components = chain[s]
            
            # the component labels in the relevant iteration
            C = self.chains['C'][s]
            
            # mapping of surface label and name
            #surfs = {0: '0', 1: 'MF', 2: 'FM'}
            
            surfs = {'0': 0, 'MF': 1, 'FM': 2}
            name = param[10:]
            c = surfs[name]
            
            
            # then plot the corresponding component
            data = getPoints(self.E)[C==c,:]
            
            if data.shape[0] == 0:
                pass
            else:
                weights = self.chains['weight'+name][s]
                # make the corresponding plot
                plotSurface(data, name, weights, components, savepath)
            
#            for c in range(3):
#                # get the relevant data
##                if c%2 == 0:
##                    data = getPoints(self.E)[C==c,:]
##                else:
##                    # if FM surface, need to reverse age pair order
##                    data = getPoints(self.E)[C==c,:][:,(1,0)]
#                # above: taken out 10/26/2020 because there has been no flip for the FM at all...
#                    
#                data = getPoints(self.E)[C==c,:]
#                
#                # 01/09/2021: skip a surface if no point is allocated on it
#                if data.shape[0] == 0:
#                    continue
#                    
#                name = surfs[c]
#                weights = self.chains['weight'+name][s]
#                
#                # make the corresponding plot
#                plotSurface(data, name, weights, components, savepath)
            
        elif param=="C":
            # s: can serve as the starting point for querying the chain
            
            # 10/11/2020: adapted to the 3-surface case
            
            def tabulate(C):
                counts = np.empty(shape=3)
                for k in range(3):
                    counts[k] = np.sum(C==k)
                return counts
            
            if s is None or s<0 or s>=len(self.chains['C']):
                s = 0
                
            Cs = np.array(self.chains['C'][s:])
            all_counts = np.apply_along_axis(tabulate, 1, Cs)
            Counts_mean = np.mean(all_counts,axis=0)
            Counts_std = np.std(all_counts,axis=0)
            
            ind = np.arange(len(Counts_mean))
            plt.bar(ind, Counts_mean, 0.5, yerr = Counts_std,
                    error_kw=dict(lw=3, capsize=3, capthick=2))
            plt.title('Number of points allocated to each type through the chain')
            plt.xticks(ind, ('0', 'MF', 'FM'))
            if savepath is not None:
                plt.savefig(savepath)
            plt.show()
            
            
        elif param.startswith('weight'):
            chain = np.array(self.chains[param])
            for k in range(self.Kmax):
                #this_label = 'comp '+str(k)
                this_label = str(k)
                plt.plot(chain[:,k],"-",label=this_label)
            #plt.legend(loc='upper right')
            plt.xlabel('Samples')
            plt.title('Traceplot of {}'.format(param))
            plt.legend()
            if savepath is not None:
                plt.savefig(savepath)
            plt.show()
            
            
        elif param.startswith('prob'):
            chain = np.array(self.chains[param])
            for h in range(3):
                this_label = str(h)
                plt.plot(chain[:,h],"-",label=this_label)
            plt.legend(loc='upper right')
            plt.xlabel('Samples')
            plt.title('Traceplot of surface allocation probs')
            if savepath is not None:
                plt.savefig(savepath)
            plt.show()
            
        else:
            plt.plot(self.chains[param])
            if savepath is not None:
                plt.savefig(savepath)
            
            plt.title('Traceplot of {}'.format(param))
            plt.xlabel('Samples')
            if savepath is not None:
                plt.savefig(savepath)
            plt.show()
            
        return
    
    def getMeanSurface(self, st=200, en=2000, thin=1, m=15, M=50,
                       plot=True, savepath=None):
        '''
        function to obtain the "mean" surface density for each surface
        '''
        
        # mapping of surface label and name
        surfs = {0: '0', 1: 'MF', 2: 'FM'}
        
        # point grids
        x = np.linspace(m, M)
        y = np.linspace(m, M)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
            
        # 01/09/2021 change: query components with type 
        # (componentsMF, _FM, _0, etc. instead of shared components)
        def calDensity(s, which):
            '''
            get the surface density (in log) at iteration s, for type which(=0,1,2)
            '''
            weights = self.chains['weight'+surfs[which]][s]
            components = self.chains['components'+surfs[which]][s]
            
            Z = evalDensity(XX, weights, components, log=True)
            Z = Z.reshape(X.shape)
            
            return Z
        
        for c in [0,1,2]:
            # go through the selected iterations and accumulate
            for s in range(st,en,thin):
                Z = calDensity(s,c)
                if s== st:
                    dens_c = Z
                else:
                    dens_c = np.concatenate((dens_c, Z), axis=0)
            # change shape       
            dens_c = dens_c.reshape(len(range(st,en,thin)),Z.shape[0],Z.shape[1])
            
            # get mean and variance
            dens_mean = np.apply_along_axis(np.mean, 0, dens_c)
            dens_sd = np.apply_along_axis(np.std, 0, dens_c)
            
            # plot
            if plot:
                ## mean
                plt.contourf(X,Y,dens_mean)
                plt.title('posterior mean log-density of the {} surface'.format(surfs[c]))
                plt.xlabel('male age')
                plt.ylabel('female age')
                if savepath is not None:
                    plt.savefig(savepath+surfs[c]+'_mean_density.pdf')
                plt.show()
                
                ## sd
                plt.contourf(X,Y,dens_sd)
                plt.title('posterior log-density std of the {} surface'.format(surfs[c]))
                plt.xlabel('male age')
                plt.ylabel('female age')
                if savepath is not None:
                    plt.savefig(savepath+surfs[c]+'_density_std.pdf')
                plt.show()
                
        return
    


#%%
# setup for simulation experiments
import pickle as pkl

import sys

os.chdir('/hpc/home/fb75/trans_flow_v3/')

array_id = sys.argv[1]

seed = int(array_id)


#%%
# useful functions

# (1) function to plot proportions of allocations (and show MF vs FM)
# with 95% credible intervals
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

#%%

# simulation experiment settings
    
# Do the two settings 100 times each
        
# priors
Pr = {"gammaScore": {'nu0': 2, 'sigma0': 1},
      "muGMM": {'mean': np.array([0,0]), 'precision': np.eye(2)*.0001,
                'covariance': np.eye(2)*10000},
      "precisionGMM": {'df': 2, 'invScale': np.eye(2), 'Scale': np.eye(2)},
      #"weight": np.ones(K), 
      "probs": np.ones(3),
      "gammaPP": {'n0': 1, 'b0': 0.02},
      #"alpha": {'a': 2.0, 'b':3.0}}   
      # 10/26/2020: try a prior for alpha to encourage less shrinkage (large alpha)
      "alpha": {'a': 4.0, 'b': 1.0}} 
    

# 01/11/2022 change number of total events in data as well
# to see if precision improves with more data
# the total number of "events" in data
      
if seed <= 200:
    N = 100
elif seed <= 400:
    N = 200
elif seed <= 600:
    N = 600
else:
    N = 800

if (seed % 200 >= 1) and (seed % 200 <= 100):
    
    # V1: equal portion of MF and FM, LESS transmission from younger men (~25) than older men (~35)
    ## 3/4 true events; 50-50 FM-MF
    ver = 'v1'
    N_non = N//4
    N_MF = (N - N_non)//2
    N_FM = N_MF
    Settings = {'N_MF': N_MF, 'N_FM': N_FM,
                'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
                'gammaL': 1, 'gammaD': 1, 
                'weightMF': np.array([0.6, 0.3, 0.05, 0.05, 0]), 
                'weightFM': np.array([0.05, 0.05, 0.6, 0, 0.3]),
                'weight0': np.array([0.05, 0.05, 0.05, 0.8, 0.05]),
                'components': [([35,20], np.diag([9,4])), ([25,20], np.diag([4,4])), 
                               ([40,35], np.diag([4,9])), ([30,30],np.diag([25,25])),
                               ([40,25],np.diag([9,9]))]}
else:
    
    # V2: 60-40 MF and FM, MORE transmissions from youner men (~25) than older men (~35)
    ## 3/4 true events; 60-40 MF-FM
    ver = 'v2'
    N_non = N//4
    N_MF = int((N - N_non) * 0.6)
    N_FM = int((N - N_non) * 0.4)
    Settings = {'N_MF': N_MF, 'N_FM': N_FM,
                 'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
                 'gammaL': 1, 'gammaD': 1, 
                 'weightMF': np.array([0.3, 0.6, 0.05, 0.05, 0]), 
                 'weightFM': np.array([0.05, 0.05, 0.6, 0, 0.3]),
                 'weight0': np.array([0.05, 0.05, 0.05, 0.8, 0.05]),
                 'components': [([35,20], np.diag([9,4])), ([25,20], np.diag([4,4])), 
                                ([40,35], np.diag([4,9])), ([30,30],np.diag([25,25])),
                                ([45,35],np.diag([9,9]))]}
    

#%%

# Simulate data
np.random.seed(seed + 367)

E, L, D = simulateLatentPoissonHGMM(N, Settings)   

#%%
## visualize a bit
#E_MF = {i:a for i,a in E.items() if i in range(150)}
#E_FM = {i:a[::-1] for i,a in E.items() if i in range(150,300)}
#
## visualize a bit
#X = getPoints(E)
#plt.plot(X[:,0], X[:,1], "o")
#plt.show()
#
#X_MF = getPoints(E_MF)
#plt.plot(X_MF[:,0], X_MF[:,1], "o")
#plt.xlim((15,50))
#plt.ylim((15,50))
#plt.title('MF surface ground truth')
#plt.show()
#
#X_FM = getPoints(E_FM)
#plt.plot(X_FM[:,0], X_FM[:,1], "o")
#plt.xlim((15,50))
#plt.ylim((15,50))
#plt.title('FM surface ground truth')
#plt.show() 

#%%

# Inference

# 01/27/2022: LOCAL RUN
# to get 3000 iters traceplot

model = LatentPoissonDPHGMM(Priors = Pr, K=3, Kmax = 10)

#model.fit(E, L, D, samples=1500, burn=500, random_seed = seed, debugHack=False)

model.fit(E, L, D, samples=3000, burn=0, random_seed = seed, debugHack=False)

#model.plotChains('N_MF')
#model.plotChains('N_FM')
#model.plotChains('muD')
#model.plotChains('muNegD')
model.plotChains('weightMF')
#model.plotChains('weightFM')
##model.plotChains('weight0')
#model.plotChains('muL')
#model.plotChains('gammaD')
#model.plotChains('gammaL')
model.plotChains('probs')
#model.plotChains('alpha_MF')
#
#model.plotChains('C', s=500)
#
##model.plotChains('componentsMF', s=1000)
#
## plot the components at MAP
#model.plotChains('componentsMF', s=np.argmax(model.chains['loglik']))
#model.plotChains('componentsFM', s=np.argmax(model.chains['loglik']))        


#%%

# save things to pickle

# 01/11/2022: update file names as well

flabel = str(array_id)+'_'+str(N)+'_'+ver

pkl.dump(model, file=open("model_"+flabel+".pkl",'wb'))

pkl.dump(model.chains['weightMF'], file = open("weightMF_"+flabel+".pkl","wb"))
pkl.dump(model.chains['C'], file = open("C_"+flabel+".pkl","wb"))



#%%

#plotProps(model.chains['C'][1000:], savepath='props_MFFM_equal.pdf')
#plotWeights(model.chains['weightMF'], savepath='younger_older_men_more_old.pdf')