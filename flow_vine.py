# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:41:19 2020

@author: lazar
"""
import numpy as np


import sys
sys.path.append(r"C:\Users\lazar\Desktop\PhD\data")
import nsf_unif as nsf
import calc_cdpdf as cd_pr
from nprm_corr import kendall_matrix
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

import edistr_funcs as edf
from nprm_copula import non_prm_dens
# from KS2D import ks2d2s

class flow_mixed_vine():
    
    def __init__(self, n_dims, z_inf):
        """
        Arguments:
            n_dims = number of dimensions of the vine model
            
        """    
        self.n_dims = n_dims
        self.z_inf = z_inf
        
    def fit_flow_margin(self,samples,n_layers,n_units,is_continuous,
                        dropout_pr=0,cond_samples=None):
        '''

        Parameters
        ----------
        samples : TYPE
            DESCRIPTION.
        n_features : TYPE
            DESCRIPTION.
        n_layers : TYPE
            DESCRIPTION.
        n_units : TYPE
            DESCRIPTION.

        Returns
        -------
        flow_output : TYPE
            DESCRIPTION.

        '''
        if samples.size>100:
            n_samp=samples.shape[0]
            samples=[samples]
            n_conds= 1
        else:
            n_samp=samples[0].shape[0]
            n_conds=samples.shape[0]
            
        flow_samples=np.empty(n_conds,dtype=object)
        copula=np.empty(n_conds,dtype=object)
        for i in range(n_conds):
            flow = nsf.ns_flow(n_features=1,n_layers=n_layers,
                              n_units=n_units,dropout_pr=dropout_pr,
                              is_continuous=is_continuous)

            flow.train(samples[i])

            if np.all(cond_samples==None):
                x_base= np.random.uniform(0,1,(n_samp,1))
                flow_samples[i]=flow.sample(samples[i],x_base)
                copula[i]= flow.transform_to_base(samples[i])
            else:
                x_base=cond_samples
                flow_samples[i]=flow.sample(samples[i],x_base)
                copula[i]= flow.transform_to_base(flow_samples[i])
                
        if n_conds>1:
            # for j in range(len(copula)):
            #     plt.figure(15)
            #     plt.subplot(5,5,j+1)
            #     plt.hist(copula[j],bins=20)
            #     copula_tf=edf.distr_tf(flow_samples[j],copula[j])
            #     plt.figure(16)
            #     plt.subplot(5,5,j+1)
            #     plt.scatter(samples[j],copula[j])
            #     plt.figure(17)
            #     plt.subplot(5,5,j+1)
            #     plt.hist(copula_tf)
            # plt.show()
            
            copula=np.concatenate(copula)
            flow_samples=np.concatenate(flow_samples)
            np.random.seed(0)
            rnd_idx=np.random.permutation([x for x in range(len(copula))])
            flow_samples=flow_samples[rnd_idx[:n_samp]]
            
            
            if is_continuous==False:
                copula_tf=edf.distr_tf(flow_samples,copula)
            else:
                copula_tf=copula
            copula=copula[rnd_idx[:n_samp]]
            # plt.figure()
            # plt.hist(copula,bins=20)
            # plt.show()
            
            # plt.figure()
            # plt.hist(copula[rnd_idx[:n_samp]])
            # plt.show()
        else:
            # Distributional transform in the discrete case
            if is_continuous==False:
                copula=edf.distr_tf(flow_samples[0],copula[0])
            else:
                copula=copula[0]
            flow_samples=flow_samples[0]
        
        return flow_samples, copula
    
    def merge_cpmf(samples, sort=False):
        flat_array=np.concatenate(samples)
        
    
    def fit_flow_cond_margin(self,samples,n_layers,n_units,is_continuous,
                        cond_samples=None,dropout_pr=0):
        '''

        Parameters
        ----------
        samples : TYPE
            DESCRIPTION.
        n_features : TYPE
            DESCRIPTION.
        n_layers : TYPE
            DESCRIPTION.
        n_units : TYPE
            DESCRIPTION.

        Returns
        -------
        flow_output : TYPE
            DESCRIPTION.

        '''
        n_samp=samples.shape[0]
        flow= nsf.ns_flow(n_features=1,n_layers=n_layers,
                       n_units=n_units,dropout_pr=dropout_pr,
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

        Parameters
        ----------
        samples : TYPE
            DESCRIPTION.
        n_features : TYPE
            DESCRIPTION.
        n_layers : TYPE
            DESCRIPTION.
        n_units : TYPE
            DESCRIPTION.

        Returns
        -------
        flow_output : TYPE
            DESCRIPTION.

        '''
        n_samp=samples.shape[0]
        flows= nsf.ns_flow(n_features=2,n_layers=1,
                       n_units=16,dropout_pr=dropout_pr,
                       res_blocks=2,n_bins=3,
                       is_continuous=True)
        flows.train(samples)
        # cfg=flow.train_CV(samples,50)
        
        # flows=np.empty(10,dtype=object)
        # for i in range(10):
        #     flow= nsf.ns_flow(n_features=2,n_layers=int(cfg[0]),
        #                n_units=int(cfg[1]),dropout_pr=dropout_pr,
        #                res_blocks=int(cfg[2]),
        #                n_bins=int(cfg[3]),is_continuous=True)
        #     flow.train(samples)
        #     flows[i]=flow
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
        
        tau_corr=kendall_matrix(samples)
        tau_sums=np.sum(np.abs(tau_corr),axis=1)
        sorted_taus=np.sort(tau_sums)
        sorted_taus=sorted_taus[::-1]
        sort_idx=np.argsort(tau_sums)
        sort_idx=sort_idx[::-1]
        
        return samples[sort_idx,:], sort_idx
    
    def tau_med(self,samples):
        tau_corr=kendall_matrix(samples)
        tau_corr_0=tau_corr-np.identity(tau_corr.shape[0])
        
    
    def build_Cvine(self,samples):
        
        # flow parameters
        n_layers=6
        n_units=16
        is_continuous=False
        
        # initialize arrays
        n_cops=int((self.n_dims**2-self.n_dims)/2)
        margins=np.empty((self.n_dims+n_cops-1), dtype=object)
        real_margs=np.empty((self.n_dims+n_cops-1), dtype=object)
        pit_margins=np.empty((self.n_dims), dtype=object)
        copulas=np.empty((n_cops), dtype=object)
        emp_copulas=np.empty((n_cops), dtype=object)
        cop_dens=np.empty((self.n_dims,self.n_dims), dtype=object)
        
        cond_margins=np.empty((self.n_dims-2,self.n_dims-1),dtype=object)
        cond_pit_margins=np.empty((self.n_dims-2,self.n_dims-1),dtype=object)
                
        # sort variables according to sums of kendall taus
        sorted_samples, sort_idx= self.sort_variables(samples)
        sorted_samples = samples
        # fit margins first and get PIT data
        for i in range(self.n_dims):
            margins[i],pit_margins[i]=self.fit_flow_margin(sorted_samples[i,:],n_layers,
                                       n_units,is_continuous)
            real_margs[i]=sorted_samples[i,:]
            print('Margin '+str(i)+' done')    
        # tree index that increases from top to bottom tree
        tree_idx=1
        cop_count=0
        marg_count=self.n_dims
        while tree_idx<self.n_dims:
            var_set= [x for x in range(tree_idx,self.n_dims)]
            if tree_idx==1:
                for i in range(len(var_set)):
                    # calculate conditional pmfs
                    cond_margins[tree_idx-1,i]=cd_pr.c_pmf(margins[var_set[i]],
                                          margins[tree_idx-1],y_cdval=1,is_continuous=is_continuous)
                    margins[marg_count],cond_pit_margins[tree_idx-1,i]=self.fit_flow_margin(
                        cond_margins[tree_idx-1,i],n_layers,n_units,is_continuous,pit_margins[var_set[i]])
                    marg_count+=1
                    real_margs[marg_count]=cond_margins[tree_idx-1,i]
                    #train copulas of 1st tree
                    copula_train=np.hstack((pit_margins[tree_idx-1],
                                            pit_margins[var_set[i]]))
                    emp_copulas[cop_count]=copula_train
                    copulas[cop_count]=self.fit_flow_copula(copula_train,n_layers,n_units)[0]
                    cop_count +=1
                print('Tree '+str(tree_idx)+' done')
                tree_idx +=1
            else:
                for i in range(len(var_set)):
                    
                    copula_train=np.hstack((cond_pit_margins[tree_idx-2,tree_idx-2],
                                            cond_pit_margins[tree_idx-2,var_set[i]-1]))
                    emp_copulas[cop_count]=copula_train
                    copulas[cop_count]=self.fit_flow_copula(copula_train,n_layers,n_units)[0]
                    cop_count +=1
                print('Tree '+str(tree_idx)+' copulas done')
                
                # calculate conditional pmfs
                if len(var_set)>=2:
                    
                    for i in range(len(var_set)):    
                        cond_margins[tree_idx-1,var_set[i]-1]=cd_pr.c_pmf(cond_margins[tree_idx-2,var_set[i]-1],
                                              cond_margins[tree_idx-2,tree_idx-2],y_cdval=1)
                        margins[marg_count],cond_pit_margins[tree_idx-1,var_set[i]-1]=self.fit_flow_margin(
                            cond_margins[tree_idx-1,var_set[i]-1],n_layers,n_units,is_continuous,
                            cond_pit_margins[tree_idx-2,var_set[i]-1])
                        real_margs[marg_count]=cond_margins[tree_idx-1,var_set[i]-1]
                        marg_count+=1
                tree_idx +=1
                print('Tree '+str(tree_idx)+' conditional margins done')
        return emp_copulas, copulas, margins, sort_idx
    
    def build_allpairs(self,samples,is_continuous=False):
        
        # flow parameters
        n_layers=1
        n_units=4
        dropout_pr=0.0
        
        # uniform distr to compare copulas
        unif2d=np.random.random(size=(samples.shape[1],2))
        
        # initialize arrays
        n_cops=int((self.n_dims**2-self.n_dims)/2)
        margins=np.empty((self.n_dims), dtype=object)
        pit_margins=np.empty((self.n_dims), dtype=object)
        emp_copulas=np.empty((n_cops), dtype=object)
        copulas=np.empty((n_cops), dtype=object)
        cop_dens_flow=np.empty((n_cops), dtype=object)
        ks_pval=np.zeros(n_cops)
        
        sorted_samples, sort_idx= self.sort_variables(samples)
        # sorted_samples = samples
        # sort_idx=[x for x in range(5)]
        # fit margins first and get PIT data
        for i in range(self.n_dims):
            margins[i],pit_margins[i]=self.fit_flow_cond_margin(sorted_samples[i,:],n_layers,
                                       n_units,is_continuous)
        
            print('Margin '+str(i)+' done')  
        # tree index that increases from top to bottom tree
        tree_idx=1
        cop_count=0
        
        while tree_idx<self.n_dims:
            var_set= [x for x in range(tree_idx,self.n_dims)]
            for i in range(len(var_set)):
                if self.z_inf==True:
                    copula_train=np.hstack((pit_margins[tree_idx-1],
                                        pit_margins[var_set[i]])) 
                else:
                    copula_train=np.hstack((pit_margins[tree_idx-1],
                                        pit_margins[var_set[i]]))   
                # emp_copulas[cop_count]=np.hstack((pit_margins[tree_idx-1],
                #                         pit_margins[var_set[i]]))
                # copulas[cop_count],cop_dens_flow[cop_count]=self.fit_flow_copula(
                #     copula_train,
                #     n_layers,n_units,
                #     dropout_pr)
                
                # ks_pval[cop_count]=ks2d2s(copula_train,unif2d)[1]
                cop_hist=np.histogram2d(x=copula_train[:,0],
                        y=copula_train[:,1],bins=[100,100])
                copulas[cop_count]=gaussian_filter(cop_hist[0], sigma=12)
                
                cop_count +=1
            tree_idx +=1
            print('Tree '+str(tree_idx)+' done')
        return emp_copulas, copulas, cop_dens_flow, margins, sort_idx,ks_pval
    
    def build_Cvine_allcd(self,samples, is_continuous=False):
        
        # flow parameters
        n_layers=1
        n_units=4
        dropout_pr=0.0
        
        # initialize arrays
        n_cops=int((self.n_dims**2-self.n_dims)/2)
        margins=np.empty((self.n_dims+n_cops-1), dtype=object)
        real_margs=np.empty((self.n_dims+n_cops-1), dtype=object)
        pit_margins=np.empty((self.n_dims), dtype=object)
        copulas=np.empty((n_cops), dtype=object)
        emp_copulas=np.empty((n_cops), dtype=object)
        cop_dens_flow=np.empty((n_cops), dtype=object)
        
        cond_margins=np.empty((self.n_dims-2,self.n_dims-1),dtype=object)
        cond_pit_margins=np.empty((self.n_dims-2,self.n_dims-1),dtype=object)
        # cond_pmf =np.empty((self.n_dims-2,self.n_dims-1,30),dtype=object)       
        # sort variables according to sums of kendall taus
        # sorted_samples, sort_idx= self.sort_variables(samples)
        sorted_samples = samples
        sort_idx=[x for x in range(5)]
        # fit margins first and get PIT data
        for i in range(self.n_dims):
            margins[i],pit_margins[i]=self.fit_flow_cond_margin(sorted_samples[i,:],n_layers,
                                       n_units,is_continuous)
        
            print('Margin '+str(i)+' done')  
        # tree index that increases from top to bottom tree
        tree_idx=1
        cop_count=0
        marg_count=self.n_dims
        while tree_idx<self.n_dims:
            var_set= [x for x in range(tree_idx,self.n_dims)]
            if tree_idx==1:
                for i in range(len(var_set)):
                    
                    print('var= '+str(i))
                    # plt.figure()
                    # if is_continuous==True:
                    #     cond_loop_range=cond_margins.shape[2]
                    # else:
                    #     cond_loop_range=len(np.unique(margins[tree_idx-1]))
                    # for j in range(cond_loop_range):
                        
                        
                    #train copulas of 1st tree
                    copula_train=np.hstack((pit_margins[tree_idx-1],
                                            pit_margins[var_set[i]]))
                    emp_copulas[cop_count]=copula_train
                    
                    
                    
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
                        cond_vals=np.histogram(margins[tree_idx-1])[1][:-1]
                    else:
                        cond_vals=np.unique(margins[tree_idx-1])
                    x_vals=np.unique(margins[var_set[i]])
                    while len(cond_vals)>cond_cdf.shape[0]:
                        cond_cdf=np.vstack((cond_cdf,cond_cdf[-1,:]))

                    # conditional margin from inv_cdf
                    marg=margins[tree_idx-1]
                    
                    print('Cond_cdf: '+str(cond_cdf.shape))
                    print('Cond_vals: '+str(cond_vals.shape))
                    print('x_vals: '+str(x_vals.shape))
                    cd_marg=[]
                    for i_val in range(len(cond_vals)):
                        cd_marg.append(edf.eval_inv_cdf(x_vals,cond_cdf[i_val,:],
                                                        emp_copulas[cop_count][np.squeeze(marg)==cond_vals[i_val],1]))
                    cd_marg=np.concatenate(cd_marg)
                    
                    print('Cd marg: '+str(cd_marg.shape)) 
                    cond_margins[tree_idx-1,i]=cd_marg
                    # margins[marg_count],cond_pit_margins[tree_idx-1,var_set[i]-1]=self.fit_flow_cond_margin(
                    #     cd_marg,n_layers,n_units,is_continuous,
                    #     cond_pit_margins[tree_idx-2,var_set[i]-1])
                    # real_margs[marg_count]=cond_margins[tree_idx-1,var_set[i]-1]
                    # marg_count+=1
                    margins[marg_count],cond_pit_margins[tree_idx-1,i]=self.fit_flow_cond_margin(
                        cd_marg,n_layers,n_units,is_continuous,pit_margins[var_set[i]])
                
                    real_margs[marg_count]=cond_margins[tree_idx-1,i]    
                    marg_count+=1

                    cop_count +=1
                print('Tree '+str(tree_idx)+' done')
                tree_idx +=1
            else:
                for i in range(len(var_set)):
                    # for j in range(len(np.unique(margins[tree_idx-1]))):
                    #     if (cond_pit_margins[tree_idx-2,tree_idx-2] is not None 
                    #         and cond_pit_margins[tree_idx-2,var_set[i]-1] is not None):
                    #         if np.any(np.logical_and(~np.isnan(cond_pit_margins[tree_idx-2,tree_idx-2]),
                    #                                  ~np.isnan(cond_pit_margins[tree_idx-2,var_set[i]-1]))):
                    copula_train=np.hstack((cond_pit_margins[tree_idx-2,tree_idx-2],
                                            cond_pit_margins[tree_idx-2,var_set[i]-1]))
                    emp_copulas[cop_count]=copula_train
                    # cop_kl[cop_count]=nn_kl(copula_train,
                    #                           real_cops[cop_count])
                    # plt.subplot(4,6,j+1)
                    # sns.jointplot(copula_train[:,0],copula_train[:,1])
                    # plt.suptitle('Copulas: '+str(i)+' Tree: '+str(tree_idx))        
                    # plt.figure()
                    # plt.scatter(cop_kl[cop_count,:],marg_kl[tree_idx-2,var_set[i],:])
                    # plt.xlabel('copula KL div')
                    # plt.ylabel('margin KL div')
                    # plt.show()
                    copulas[cop_count],cop_dens_flow[cop_count]=self.fit_flow_copula(copula_train,
                                                                                      n_layers,n_units,
                                                                                      dropout_pr)
                    
                    # calculate conditional pmfs
                    if len(var_set)>=2:
                        cond_pmf=cd_pr.c_pmf(cond_margins[tree_idx-2,var_set[i]-1],
                                                                            cond_margins[tree_idx-2,tree_idx-2],
                                                                            x_continuous=is_continuous,
                                                                            y_continuous=is_continuous)
                        cond_pmf=np.array([cond_pmf[x,:]/np.sum(cond_pmf[x,:]) for x in range(
                                cond_pmf.shape[0]-1)])
                        cond_cdf=np.cumsum(cond_pmf,axis=1)
                        if is_continuous:
                            cond_vals=np.histogram(cond_margins[tree_idx-2,tree_idx-2])[1][:-1]
                        else:
                            cond_vals=np.unique(cond_margins[tree_idx-2,tree_idx-2])
                        x_vals=np.unique(cond_margins[tree_idx-2,var_set[i]-1])
                        
                        while len(cond_vals)>cond_cdf.shape[0]:
                            cond_cdf=np.vstack((cond_cdf,cond_cdf[-1,:]))
                         
                        print('Cond_cdf: '+str(cond_cdf.shape))
                        print('Cond_vals: '+str(cond_vals.shape))
                        print('x_vals: '+str(x_vals.shape))
                        print('emp_cop: '+str(copula_train))
                        # conditional margin from inv_cdf
                        marg=cond_margins[tree_idx-2,tree_idx-2]
                        cd_marg=[]
                        for i_val in range(len(cond_vals)):
                            cd_marg.append(edf.eval_inv_cdf(x_vals,cond_cdf[i_val,:],
                                                            emp_copulas[cop_count][np.squeeze(marg)==cond_vals[i_val],1]))
                        cd_marg=np.concatenate(cd_marg)
                        print('Cd marg: '+str(cd_marg.shape))
                            
                        cond_margins[tree_idx-1,var_set[i]-1]=cd_marg
                        margins[marg_count],cond_pit_margins[tree_idx-1,var_set[i]-1]=self.fit_flow_cond_margin(
                            cd_marg,n_layers,n_units,is_continuous,
                            cond_pit_margins[tree_idx-2,var_set[i]-1])
                        real_margs[marg_count]=cond_margins[tree_idx-1,var_set[i]-1]
                        marg_count+=1
                        plt.suptitle('Conditional margins: '+str(var_set[i])
                                 +' Tree: '+str(tree_idx))
                    cop_count +=1
                print('Tree '+str(tree_idx)+' copulas done')
                
                # calculate conditional pmfs
                # if len(var_set)>=2:
                    
                #     for i in range(len(var_set)):
                #         cond_pmf=cd_pr.c_pmf(cond_margins[tree_idx-2,var_set[i]-1],
                #                                                             cond_margins[tree_idx-2,tree_idx-2],
                #                                                             x_continuous=is_continuous,
                #                                                             y_continuous=is_continuous)
                #         cond_pmf=np.array([cond_pmf[x,:]/np.sum(cond_pmf[x,:]) for x in range(
                #                 cond_pmf.shape[0]-1)])
                #         cond_cdf=np.cumsum(cond_pmf[:,:-1],axis=1)
                #         if is_continuous:
                #             cond_vals=np.histogram(cond_margins[tree_idx-2,tree_idx-2])[1][:-1]
                #         else:
                #             cond_vals=np.unique(cond_margins[tree_idx-2,tree_idx-2])
                #         x_vals=np.unique(cond_margins[tree_idx-2,var_set[i]-1])
                        
                #         # conditional margin 2|1 from inv_cdf
                #         marg=cond_margins[tree_idx-2,var_set[i]-1]
                #         cd_marg=[]
                #         for i in range(len(cond_vals)):
                #             cd_marg.append(edf.eval_inv_cdf(x_vals,cond_cdf[i,:],
                #                                             emp_cop[1][0,marg==cond_vals[i]]))
                #         cd_marg=np.concatenate(cd_marg)
                #         if len(x_vals)>cond_cdf.shape[0]:
                #             cond_cdf=np.vstack((cond_cdf,cond_cdf[-1,:]))
                #         qq,ccdf=edf.ecdf(cd_marg21)
                #         cop21= edf.eval_cdf(ccdf,qq[:-1],cd_marg21)
                #         cop21= edf.distr_tf(cd_marg21,cop21).T
                            
                #         cond_margins[tree_idx-1,var_set[i]-1]
                #         margins[marg_count],cond_pit_margins[tree_idx-1,var_set[i]-1]=self.fit_flow_cond_margin(
                #             cond_margins[tree_idx-1,var_set[i]-1],n_layers,n_units,is_continuous,
                #             cond_pit_margins[tree_idx-2,var_set[i]-1])
                #         real_margs[marg_count]=cond_margins[tree_idx-1,var_set[i]-1]
                #         marg_count+=1
                #         plt.suptitle('Conditional margins: '+str(var_set[i])
                #                  +' Tree: '+str(tree_idx))
                tree_idx +=1
                print('Tree '+str(tree_idx)+' conditional margins done')
        return emp_copulas, copulas, cop_dens_flow, margins,real_margs, sort_idx
    
    def build_Cvine_all_npr(self,samples, is_continuous=False):
        
        # flow parameters
        n_layers=1
        n_units=8
        
        
        # initialize arrays
        n_cops=int((self.n_dims**2-self.n_dims)/2)
        margins=np.empty((self.n_dims+n_cops-1), dtype=object)
        real_margs=np.empty((self.n_dims+n_cops-1), dtype=object)
        pit_margins=np.empty((self.n_dims), dtype=object)
        copulas=np.empty((n_cops), dtype=object)
        emp_copulas=np.empty((n_cops), dtype=object)
        cop_dens_flow=np.empty((n_cops), dtype=object)
        cop_dens_tll1=np.empty((n_cops), dtype=object)
        cop_dens_tll2=np.empty((n_cops), dtype=object)
        cop_dens_tll1nn=np.empty((n_cops), dtype=object)
        cop_dens_tll2nn=np.empty((n_cops), dtype=object)
        cop_dens_bern=np.empty((n_cops), dtype=object)

        cond_margins=np.empty((self.n_dims-2,self.n_dims-1),dtype=object)
        cond_pit_margins=np.empty((self.n_dims-2,self.n_dims-1),dtype=object)
        # cond_pmf =np.empty((self.n_dims-2,self.n_dims-1,30),dtype=object)       
        # sort variables according to sums of kendall taus
        # sorted_samples, sort_idx= self.sort_variables(samples)
        sorted_samples = samples
        # fit margins first and get PIT data
        for i in range(self.n_dims):
            margins[i],pit_margins[i]=self.fit_flow_cond_margin(sorted_samples[i,:],n_layers,
                                       n_units,is_continuous)
        
            print('Margin '+str(i)+' done')  
        # tree index that increases from top to bottom tree
        tree_idx=1
        cop_count=0
        marg_count=self.n_dims
        while tree_idx<self.n_dims:
            var_set= [x for x in range(tree_idx,self.n_dims)]
            if tree_idx==1:
                for i in range(len(var_set)):
                    # calculate conditional pmfs
                    print('var= '+str(i))

                    cond_margins[tree_idx-1,i]=cd_pr.c_pmf(margins[var_set[i]],
                                          margins[tree_idx-1],is_continuous=is_continuous)

                    temp_cond_marg=cond_margins[tree_idx-1,i]

                    margins[marg_count],cond_pit_margins[tree_idx-1,i]=self.fit_flow_cond_margin(
                        temp_cond_marg,n_layers,n_units,is_continuous,pit_margins[var_set[i]])
                
                    real_margs[marg_count]=cond_margins[tree_idx-1,i]    
                    marg_count+=1


                    #train copulas of 1st tree
                    copula_train=np.hstack((pit_margins[tree_idx-1],
                                            pit_margins[var_set[i]]))
                    emp_copulas[cop_count]=copula_train
                    # sns.jointplot(copula_train[:,0],copula_train[:,1])
                    
                    
                    # cop_dens_flow[cop_count]=self.fit_flow_copula(copula_train,n_layers,n_units)
                    
                    cop_dens_tll1[cop_count]=non_prm_dens(copula_train,method="TLL1",knots=30)
                    cop_dens_tll2[cop_count]=non_prm_dens(copula_train,method="TLL2",knots=30)
                    cop_dens_tll1nn[cop_count]=non_prm_dens(copula_train,method="TLL1nn",knots=30)
                    cop_dens_tll2nn[cop_count]=non_prm_dens(copula_train,method="TLL2nn",knots=30)
                    cop_dens_bern[cop_count]=non_prm_dens(copula_train,method="bern",knots=30)
                    
                    cop_count +=1
                print('Tree '+str(tree_idx)+' done')
                tree_idx +=1
            else:
                for i in range(len(var_set)):
                    
                    copula_train=np.hstack((cond_pit_margins[tree_idx-2,tree_idx-2],
                                            cond_pit_margins[tree_idx-2,var_set[i]-1]))
                    emp_copulas[cop_count]=copula_train
                    # cop_dens_flow[cop_count]=self.fit_flow_copula(copula_train,n_layers,n_units)
                    cop_dens_tll1[cop_count]=non_prm_dens(copula_train,method="TLL1",knots=30)
                    cop_dens_tll2[cop_count]=non_prm_dens(copula_train,method="TLL2",knots=30)
                    cop_dens_tll1nn[cop_count]=non_prm_dens(copula_train,method="TLL1nn",knots=30)
                    cop_dens_tll2nn[cop_count]=non_prm_dens(copula_train,method="TLL2nn",knots=30)
                    cop_dens_bern[cop_count]=non_prm_dens(copula_train,method="bern",knots=30)
                    cop_count +=1
                print('Tree '+str(tree_idx)+' copulas done')
                
                # calculate conditional pmfs
                if len(var_set)>=2:
                    
                    for i in range(len(var_set)):
                        cond_margins[tree_idx-1,var_set[i]-1]=cd_pr.c_pmf(cond_margins[tree_idx-2,var_set[i]-1],
                                                                            cond_margins[tree_idx-2,tree_idx-2],
                                                                            is_continuous=is_continuous)
                        margins[marg_count],cond_pit_margins[tree_idx-1,var_set[i]-1]=self.fit_flow_cond_margin(
                            cond_margins[tree_idx-1,var_set[i]-1],n_layers,n_units,is_continuous,
                            cond_pit_margins[tree_idx-2,var_set[i]-1])
                        real_margs[marg_count]=cond_margins[tree_idx-1,var_set[i]-1]
                        marg_count+=1
                        plt.suptitle('Conditional margins: '+str(var_set[i])
                                 +' Tree: '+str(tree_idx))
                tree_idx +=1
                print('Tree '+str(tree_idx)+' conditional margins done')
        return cop_dens_flow,cop_dens_tll1,cop_dens_tll2,cop_dens_tll1nn,cop_dens_tll2nn,cop_dens_bern
    
    def build_Cvine_npr_flow(self,samples, is_continuous=False):
        
        # flow parameters
        n_layers=1
        n_units=8
        
        
        # initialize arrays
        n_cops=int((self.n_dims**2-self.n_dims)/2)
        margins=np.empty((self.n_dims+n_cops-1), dtype=object)
        real_margs=np.empty((self.n_dims+n_cops-1), dtype=object)
        pit_margins=np.empty((self.n_dims), dtype=object)
        copulas=np.empty((n_cops), dtype=object)
        emp_copulas=np.empty((n_cops), dtype=object)
        cop_dens_flow=np.empty((n_cops), dtype=object)


        cond_margins=np.empty((self.n_dims-2,self.n_dims-1),dtype=object)
        cond_pit_margins=np.empty((self.n_dims-2,self.n_dims-1),dtype=object)
        # cond_pmf =np.empty((self.n_dims-2,self.n_dims-1,30),dtype=object)       
        # sort variables according to sums of kendall taus
        # sorted_samples, sort_idx= self.sort_variables(samples)
        sorted_samples = samples
        # fit margins first and get PIT data
        for i in range(self.n_dims):
            margins[i],pit_margins[i]=self.fit_flow_cond_margin(sorted_samples[i,:],n_layers,
                                       n_units,is_continuous)
        
            print('Margin '+str(i)+' done')  
        # tree index that increases from top to bottom tree
        tree_idx=1
        cop_count=0
        marg_count=self.n_dims
        while tree_idx<self.n_dims:
            var_set= [x for x in range(tree_idx,self.n_dims)]
            if tree_idx==1:
                for i in range(len(var_set)):
                    # calculate conditional pmfs
                    print('var= '+str(i))

                    cond_margins[tree_idx-1,i]=cd_pr.c_pmf(margins[var_set[i]],
                                          margins[tree_idx-1],is_continuous=is_continuous)

                    temp_cond_marg=cond_margins[tree_idx-1,i]

                    margins[marg_count],cond_pit_margins[tree_idx-1,i]=self.fit_flow_cond_margin(
                        temp_cond_marg,n_layers,n_units,is_continuous,pit_margins[var_set[i]])
                
                    real_margs[marg_count]=cond_margins[tree_idx-1,i]    
                    marg_count+=1


                    #train copulas of 1st tree
                    copula_train=np.hstack((pit_margins[tree_idx-1],
                                            pit_margins[var_set[i]]))
                    emp_copulas[cop_count]=copula_train
                    
                    
                    
                    cop_dens_flow[cop_count]=self.fit_flow_copula(copula_train,n_layers,n_units)
                    
                    cop_count +=1
                print('Tree '+str(tree_idx)+' done')
                tree_idx +=1
            else:
                for i in range(len(var_set)):
                    
                    copula_train=np.hstack((cond_pit_margins[tree_idx-2,tree_idx-2],
                                            cond_pit_margins[tree_idx-2,var_set[i]-1]))
                    emp_copulas[cop_count]=copula_train
                    cop_dens_flow[cop_count]=self.fit_flow_copula(copula_train,n_layers,n_units)
                    cop_count +=1
                print('Tree '+str(tree_idx)+' copulas done')
                
                # calculate conditional pmfs
                if len(var_set)>=2:
                    
                    for i in range(len(var_set)):
                        cond_margins[tree_idx-1,var_set[i]-1]=cd_pr.c_pmf(cond_margins[tree_idx-2,var_set[i]-1],
                                                                            cond_margins[tree_idx-2,tree_idx-2],
                                                                            is_continuous=is_continuous)
                        margins[marg_count],cond_pit_margins[tree_idx-1,var_set[i]-1]=self.fit_flow_cond_margin(
                            cond_margins[tree_idx-1,var_set[i]-1],n_layers,n_units,is_continuous,
                            cond_pit_margins[tree_idx-2,var_set[i]-1])
                        real_margs[marg_count]=cond_margins[tree_idx-1,var_set[i]-1]
                        marg_count+=1
                        plt.suptitle('Conditional margins: '+str(var_set[i])
                                 +' Tree: '+str(tree_idx))
                tree_idx +=1
                print('Tree '+str(tree_idx)+' conditional margins done')
        return cop_dens_flow
    
    def build_Cvine_prune(self,samples, is_continuous=False):
        
        # flow parameters
        n_layers=1
        n_units=4
        dropout_pr=0.0
        
        # initialize arrays
        n_cops=int((self.n_dims**2-self.n_dims)/2)
        margins=np.empty((self.n_dims+n_cops-1), dtype=object)
        real_margs=np.empty((self.n_dims+n_cops-1), dtype=object)
        pit_margins=np.empty((self.n_dims), dtype=object)
        copulas=np.empty((n_cops), dtype=object)
        emp_copulas=np.empty((n_cops), dtype=object)
        cop_dens_flow=np.empty((n_cops), dtype=object)
        cop_tree_idx=np.zeros((n_cops,3))
        
        cond_margins=np.empty((self.n_dims-2,self.n_dims-1),dtype=object)
        cond_pit_margins=np.empty((self.n_dims-2,self.n_dims-1),dtype=object)
      
        # sort variables according to sums of kendall taus
        sorted_samples, sort_idx= self.sort_variables(samples)
        # sorted_samples = samples
        # fit margins first and get PIT data
        for i in range(self.n_dims):
            margins[i],pit_margins[i]=self.fit_flow_cond_margin(sorted_samples[i,:],n_layers,
                                       n_units,is_continuous)
        
            print('Margin '+str(i)+' done')  
        # tree index that increases from top to bottom tree
        tree_idx=1
        cop_count=0
        marg_count=self.n_dims
        while tree_idx<self.n_dims:
            var_set= [x for x in range(tree_idx,self.n_dims)]
            if tree_idx==1:
                for i in range(len(var_set)):
                    # calculate conditional pmfs
                    print('var= '+str(i))

                    cond_margins[tree_idx-1,i]=cd_pr.c_pmf(margins[var_set[i]],
                                          margins[tree_idx-1],is_continuous=is_continuous)

                    temp_cond_marg=cond_margins[tree_idx-1,i]

                    
                    margins[marg_count],cond_pit_margins[tree_idx-1,i]=self.fit_flow_cond_margin(
                        temp_cond_marg,n_layers,n_units,is_continuous,pit_margins[var_set[i]])
                
                    real_margs[marg_count]=cond_margins[tree_idx-1,i]    
                    marg_count+=1


                    #train copulas of 1st tree
                    copula_train=np.hstack((pit_margins[tree_idx-1],
                                            pit_margins[var_set[i]]))
                    emp_copulas[cop_count]=copula_train
                    
                    cop_tree_idx[cop_count,0]=tree_idx-1
                    cop_tree_idx[cop_count,1]=var_set[i]
                    cop_tree_idx[cop_count,2]=tree_idx
                    
                    copulas[cop_count],cop_dens_flow[cop_count]=self.fit_flow_copula(copula_train,
                                                                                      n_layers,n_units,
                                                                                      dropout_pr)
                    cop_count +=1
                print('Tree '+str(tree_idx)+' done')
                tree_idx +=1
            else:
                for i in range(len(var_set)):
                   
                    copula_train=np.hstack((cond_pit_margins[tree_idx-2,tree_idx-2],
                                            cond_pit_margins[tree_idx-2,var_set[i]-1]))
                    emp_copulas[cop_count]=copula_train

                    cop_tree_idx[cop_count,0]=tree_idx-1
                    cop_tree_idx[cop_count,1]=var_set[i]
                    cop_tree_idx[cop_count,2]=tree_idx

                    copulas[cop_count],cop_dens_flow[cop_count]=self.fit_flow_copula(copula_train,
                                                                                      n_layers,n_units,
                                                                                      dropout_pr)
                    cop_count +=1
                print('Tree '+str(tree_idx)+' copulas done')
                
                # calculate conditional pmfs
                if len(var_set)>=2:
                    
                    for i in range(len(var_set)):
                        cond_margins[tree_idx-1,var_set[i]-1]=cd_pr.c_pmf(cond_margins[tree_idx-2,var_set[i]-1],
                                                                            cond_margins[tree_idx-2,tree_idx-2],
                                                                            is_continuous=is_continuous)
                        margins[marg_count],cond_pit_margins[tree_idx-1,var_set[i]-1]=self.fit_flow_cond_margin(
                            cond_margins[tree_idx-1,var_set[i]-1],n_layers,n_units,is_continuous,
                            cond_pit_margins[tree_idx-2,var_set[i]-1])
                        real_margs[marg_count]=cond_margins[tree_idx-1,var_set[i]-1]
                        marg_count+=1
                        plt.suptitle('Conditional margins: '+str(var_set[i])
                                 +' Tree: '+str(tree_idx))
                tree_idx +=1
                print('Tree '+str(tree_idx)+' conditional margins done')
        return emp_copulas, copulas, cop_dens_flow, margins,real_margs, sort_idx