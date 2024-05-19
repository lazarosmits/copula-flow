# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:41:19 2020

@author: lazar
"""
import numpy as np

import nsf_unif as nsf
import calc_cdpdf as cd_pr
from nprm_corr import kendall_matrix
import edistr_funcs as edf


'''

This module implements a copula C-Vine model that fits Neural Spline Flows (NSF)
for margins and copulas of joint continuous or discrete observations  

'''

class flow_mixed_vine():
    
    def __init__(self, n_dims):
        
        """
        
        Arguments:
            n_dims = number of dimensions of the vine model
            
        """    
        self.n_dims = n_dims        
    
    def fit_flow_cond_margin(self,samples,n_layers,n_units,is_continuous,
                        cond_samples=None,dropout_pr=0):
        '''
        fits a NSF model for a conditional margin or unconditional margin
        if the conditioning set is empty (1st tree)
        
        Parameters
        ----------
        samples : numpy array, observation data matrix, dimensions-by-samples
        n_layers : int, number of layers in the NSF model
        n_units : int, number of hidden units in the NSF model
        is_continuous : bool, True for continuous and False for discrete data
        cond_samples: numpy array, samples of margin to condition on. None if unconditional
        dropout_pr: float, [0,1] dropout probability for NSF model    

        Returns
        -------
        flow_samples : numpy array, samples of NSF conditional margin
        copula: numpy array, samples of the margin transformed to copula space

        '''
        n_samp=samples.shape[0]
        flow= nsf.ns_flow(n_features=1,n_layers=2,
                       n_units=8,dropout_pr=dropout_pr,
                       is_continuous=is_continuous)
        flow.train(samples)
        if np.all(cond_samples==None):
            x_base= np.random.uniform(0,1,(n_samp,1))
            flow_samples= flow.sample(samples,x_base)
            copula= flow.transform_to_base(samples)[0]
            if is_continuous==False:
                copula=edf.distr_tf(samples,copula)
        else:
            x_base=cond_samples
            flow_samples= flow.sample(samples,x_base)
            copula= flow.transform_to_base(flow_samples)[0]
            if is_continuous==False:
                copula=edf.distr_tf(flow_samples,copula)
        return flow_samples, copula
    
    def fit_flow_copula(self,samples,n_layers,n_units,dropout_pr=0):
        
        '''
        fits a NSF model for a bivariate copula
        
        Parameters
        ----------
        samples : numpy array, empirical copula samples, 2-by-n_samples
        n_layers : int, number of layers in the NSF model
        n_units : int, number of hidden units in the NSF model
        dropout_pr: float, [0,1] dropout probability for NSF model  

        Returns
        -------
        flow_samples : numpy array, 2-by-n_samples, samples of NSF copula
        density : numpy array, NSF copula density function on a 200-by-200 grid

        '''
        n_samp=samples.shape[0]
        flows= nsf.ns_flow(n_features=2,n_layers=1,
                       n_units=16,dropout_pr=dropout_pr,
                       res_blocks=3,n_bins=2,
                       is_continuous=True)
        flows.train(samples)

        x_base= np.random.uniform(0,1,(n_samp,2))
        flow_samples= flows.sample(samples,x_base)
        xx,yy=np.meshgrid(np.linspace(0,1,200),np.linspace(0,1,200))
        grid=np.vstack((xx.ravel(),yy.ravel()))
        density= flows.density(grid).reshape(xx.shape)

        # margin normalization step
        for i in range(1500):
            marg1=np.sum(density,axis=0)
            marg2=np.sum(density,axis=1)
            density=density/(marg1*marg2)
            density=(density/np.sum(density))
        return flow_samples,density
    
    def sort_variables(self,samples):
        
        '''
        sorts the variables in the joint observation data matrix 
        in descending order according to the sums of Kendall's taus
        
        Parameters
        ----------
        samples : numpy array, unsorted observation data matrix, 
                  dimensions-by-samples
                  
        Returns
        -------
        samples[sort_idx,:] : numpy array, sorted data matrix
        sort_idx: list, sorting index for the variables
        
        '''
        
        tau_corr=kendall_matrix(samples)
        tau_sums=np.sum(np.abs(tau_corr),axis=1)
        sorted_taus=np.sort(tau_sums)
        sorted_taus=sorted_taus[::-1]
        sort_idx=np.argsort(tau_sums)
        sort_idx=sort_idx[::-1]
        
        return samples[sort_idx,:], sort_idx
    
    def build_Cvine(self,samples, is_continuous=False):
        
        '''
        builds the C-vine and fits NSF models for the copulas and margins 
        
        Parameters
        ----------
        samples : numpy array, observation data matrix, dimensions-by-N_samples 
        is_continuous : bool, True for continuous and False for discrete data         
                  
        Returns 
        -------
        the following numpy objects with n_copulas = (dim**2-dim)/2 number 
        of entries
        
            emp_copulas : numpy arrays, empirical pair copulas in the C-vine
            copulas : numpy arrays, samples of NSF-based pair copulas in the C-vine
            cop_dens_flow : numpy arrays, NSF-based densities of 
                            pair copulas in the C-vine
        
        and also the following numpy objects with n_copulas + n_dim number 
        of entries
            margins: NSF-based margins of variables
            real_margs: empirical margins of variables
        
        sort_idx: list, sorting index for the variables
        
        '''
        
        # normalizing flow parameters
        n_layers=1
        n_units=4
        dropout_pr=0.0
        
        # number of copulas in the vine
        n_cops=int((self.n_dims**2-self.n_dims)/2)
        
        # initialize arrays
        margins=np.empty((self.n_dims+n_cops-1), dtype=object)  # flow margins
        real_margs=np.empty((self.n_dims+n_cops-1), dtype=object) # real margins
        pit_margins=np.empty((self.n_dims), dtype=object) # transformed margins
        copulas=np.empty((n_cops), dtype=object)  # flow copulas (samples)
        emp_copulas=np.empty((n_cops), dtype=object) # empirical copulas
        cop_dens_flow=np.empty((n_cops), dtype=object) # flow copula pdfs
        
        cond_margins=np.empty((self.n_dims-2,self.n_dims-1),
                              dtype=object) # conditional margins
        cond_pit_margins=np.empty((self.n_dims-2,self.n_dims-1),
                                  dtype=object) # conditional transformed margins

        sorted_samples, sort_idx= self.sort_variables(samples)

        # fit margins first and get PIT data
        for i in range(self.n_dims):
            print('Fitting Margin '+str(i+1)) 
            margins[i],pit_margins[i]=self.fit_flow_cond_margin(
                                            sorted_samples[i,:],n_layers,
                                            n_units,is_continuous)
        
             
        # tree index that increases from top to bottom tree
        tree_idx=1
        cop_count=0
        marg_count=self.n_dims
        while tree_idx<self.n_dims:
            var_set= [x for x in range(tree_idx,self.n_dims)]
            if tree_idx==1:
                for i in range(len(var_set)):
                    
                    # print('var= '+str(i))
                        
                    #train copulas of 1st tree
                    copula_train=np.hstack((pit_margins[tree_idx-1],
                                            pit_margins[var_set[i]]))
                    emp_copulas[cop_count]=copula_train
                    
                    
                    print('Fitting Copula '+str(cop_count+1)+' Tree '+str(tree_idx))
                    copulas[cop_count],cop_dens_flow[cop_count]=self.fit_flow_copula(copula_train,
                                                                                      n_layers,n_units,
                                                                                      dropout_pr)
                    
                    # calculate conditional pmfs
                    
                    cond_pmf=cd_pr.c_pmf(margins[var_set[i]],
                                                    margins[tree_idx-1],
                                                    x_continuous=is_continuous,
                                                    y_continuous=is_continuous)

                    cond_pmf=np.array([cond_pmf[x,:]/np.sum(cond_pmf[x,:]) for x in range(
                            cond_pmf.shape[0]-1)])
                    cond_cdf=np.cumsum(cond_pmf,axis=1)
                    if is_continuous:
                        cond_vals=np.histogram(margins[tree_idx-1],bins=25)[1]
                        cond_vals[-1]=cond_vals[-1]+0.0001
                    else:
                        cond_vals=np.unique(margins[tree_idx-1])
                    x_vals=np.unique(margins[var_set[i]])
                    while len(cond_vals)>cond_cdf.shape[0]:
                        cond_cdf=np.vstack((cond_cdf,cond_cdf[-1,:]))

                    # conditional margin from inv_cdf
                    marg=margins[tree_idx-1]
                    

                    cd_marg=[]
                    if is_continuous:
                        for i_val in range(len(cond_vals)-1):
                            idx_vals=np.logical_and(np.squeeze(marg)>=cond_vals[i_val],
                                                    np.squeeze(marg)<cond_vals[i_val+1])
                            cd_marg.append(edf.eval_inv_cdf(x_vals,cond_cdf[i_val,:],
                                            emp_copulas[cop_count][idx_vals,1]))
                    else:
                        for i_val in range(len(cond_vals)):
                            cd_marg.append(edf.eval_inv_cdf(x_vals,cond_cdf[i_val,:],
                                            emp_copulas[cop_count][
                                        np.squeeze(marg)==cond_vals[i_val],1]))
                    cd_marg=np.concatenate(cd_marg)
                    

                    cond_margins[tree_idx-1,i]=cd_marg

                    print('Fitting Cond_margin '+str(var_set[i])+' Tree '+str(tree_idx))
                    margins[marg_count],cond_pit_margins[tree_idx-1,i]=self.fit_flow_cond_margin(
                        cd_marg,n_layers,n_units,is_continuous,pit_margins[var_set[i]])
                
                    real_margs[marg_count]=cond_margins[tree_idx-1,i]    
                    marg_count+=1

                    cop_count +=1
                print('Tree '+str(tree_idx)+' done')
                tree_idx +=1
            else:
                for i in range(len(var_set)):
                   
                    copula_train=np.hstack((cond_pit_margins[tree_idx-2,tree_idx-2],
                                            cond_pit_margins[tree_idx-2,var_set[i]-1]))
                    emp_copulas[cop_count]=copula_train

                    print('Fitting Copula '+str(cop_count)+' Tree '+str(tree_idx))
                    copulas[cop_count],cop_dens_flow[cop_count]=self.fit_flow_copula(copula_train,
                                                                                      n_layers,n_units,
                                                                                      dropout_pr)
                    
                    # calculate conditional pmfs
                    if len(var_set)>=2:
                        
                        print('Cond_margin '+str([tree_idx-2,var_set[i]-1])+' training')
                        cond_pmf=cd_pr.c_pmf(cond_margins[tree_idx-2,var_set[i]-1],
                                                                            cond_margins[tree_idx-2,tree_idx-2],
                                                                            x_continuous=is_continuous,
                                                                            y_continuous=is_continuous)
                        cond_pmf=np.array([cond_pmf[x,:]/np.sum(cond_pmf[x,:]) for x in range(
                                    cond_pmf.shape[0]-1)])
                                
                        cond_pmf[np.isnan(cond_pmf)]=0
                        cond_cdf=np.cumsum(cond_pmf,axis=1)
                        if is_continuous:
                            cond_vals=np.histogram(cond_margins[tree_idx-2,tree_idx-2],
                                                   bins=25)[1]
                            cond_vals[-1]=cond_vals[-1]+0.0001
                        else:
                            cond_vals=np.unique(cond_margins[tree_idx-2,tree_idx-2])
                        x_vals=np.unique(cond_margins[tree_idx-2,var_set[i]-1])
                        
                        while len(cond_vals)>cond_cdf.shape[0]:
                            cond_cdf=np.vstack((cond_cdf,cond_cdf[-1,:]))
                         

                        # conditional margin from inv_cdf
                        marg=cond_margins[tree_idx-2,tree_idx-2]
                       
                        cd_marg=[]
                        if is_continuous:
                            for i_val in range(len(cond_vals)-1):
                                idx_vals=np.logical_and(np.squeeze(marg)>=cond_vals[i_val],
                                                        np.squeeze(marg)<cond_vals[i_val+1])
                                cd_marg.append(edf.eval_inv_cdf(x_vals,cond_cdf[i_val,:],
                                                emp_copulas[cop_count][idx_vals,1]))
                        else:
                            for i_val in range(len(cond_vals)):
                                cd_marg.append(edf.eval_inv_cdf(x_vals,cond_cdf[i_val,:],
                                                emp_copulas[cop_count][
                                            np.squeeze(marg)==cond_vals[i_val],1]))
                        
                        
                        
                        cd_marg=np.concatenate(cd_marg)
                        # print('Cd marg: '+str(cd_marg.shape))
                            
                        cond_margins[tree_idx-1,var_set[i]-1]=cd_marg
                        margins[marg_count],cond_pit_margins[tree_idx-1,var_set[i]-1]=self.fit_flow_cond_margin(
                            cd_marg,n_layers,n_units,is_continuous,
                            cond_pit_margins[tree_idx-2,var_set[i]-1])
                        real_margs[marg_count]=cond_margins[tree_idx-1,var_set[i]-1]
                        marg_count+=1
                    cop_count +=1
                print('Tree '+str(tree_idx)+' copulas done')
                
                tree_idx +=1
                print('Tree '+str(tree_idx)+' conditional margins done')
        return emp_copulas, copulas, cop_dens_flow, margins,real_margs, sort_idx
    

    
