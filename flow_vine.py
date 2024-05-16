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


class flow_mixed_vine():
    
    def __init__(self, n_dims):
        """
        Arguments:
            n_dims = number of dimensions of the vine model
            
        """    
        self.n_dims = n_dims
        
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

        else:
            # Distributional transform in the discrete case
            if is_continuous==False:
                copula=edf.distr_tf(flow_samples[0],copula[0])
            else:
                copula=copula[0]
            flow_samples=flow_samples[0]
        
        return flow_samples, copula
        
    
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
    
    def build_Cvine(self,samples, is_continuous=False):
        
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
    

    
