# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 03:54:31 2020

@author: lazar
"""

import numpy as np
import edistr_funcs as edf
from sample_funcs import rej_sample

def bin_2ddensity(x,x_grid,y_grid,p1,p2):
    '''
    
    Parameters
    ----------
    x : 2d density in grid.
    p1 : probabilities for quantiles in variable 1
    p2 : probabilities for quantiles in variable 2

    Returns
    -------
    binned_density : normalized counts in binned grid 

    '''
    
    
    p1_idx= np.zeros((len(p1),)).astype('int')
    p1_idx= np.hstack((0,p1_idx))
    p2_idx= np.zeros((len(p2),)).astype('int')
    p2_idx= np.hstack((0,p2_idx))
    # temporary array for averaging with variable 1 grid
    binned_temp= np.zeros((len(p1),len(y_grid)))
    for i in range(len(p1)):
        # find indices corresponding to p1 in the grid
        p1_idx[i+1]=np.argmin(abs(p1[i]-x_grid))
        # take averages of values for the specified bins
        if p1_idx[i]==p1_idx[i+1]:
            binned_temp[i,:]=x[-1,:]
        else:
            binned_temp[i,:]=np.mean(x[p1_idx[i]:p1_idx[i+1],:],axis=0)
        
    # initialize output array with both variables averaged
    binned_density= np.zeros((len(p1),len(p2)))
    for i in range(len(p2)):
        p2_idx[i+1]=np.argmin(abs(p2[i]-y_grid))
        if p2_idx[i]==p2_idx[i+1]:
            binned_density[:,i]=binned_temp[:,-1]
        else:
            binned_density[:,i]=np.mean(binned_temp[:,p2_idx[i]:p2_idx[i+1]],axis=1)
    # normalize counts of density
    binned_density=  binned_density/np.sum(np.sum(binned_density))   
    
    return binned_density
    
def h_func(x,x_grid,y_grid,p1,p2):
    
    binned_density=bin_2ddensity(x,x_grid,y_grid,p1,p2) 
    x_diff= np.gradient(x,axis=0)
    x_trans=(np.mean(x_diff,axis=0)+abs(min(np.mean(x_diff,axis=0))))
    x_trans=x_trans/max(x_trans)
    return x_trans


def c_pmf(x,y,x_continuous=False, y_continuous=False):
    '''

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    cd_pdf : TYPE
        DESCRIPTION.

    '''
    
    if not x_continuous and not y_continuous:
        # estimate the different conditional cdfs
        qx_pos,c_cdf_x_pos=edf.ecdf(x)
        qx_neg,c_cdf_x_neg=qx_pos,np.hstack((np.zeros(1,),c_cdf_x_pos[:-1]))
        
        qy_pos,c_cdf_y_pos=edf.ecdf(y)
        qy_neg,c_cdf_y_neg=qy_pos,np.hstack((np.zeros(1,),c_cdf_y_pos[:-1]))
        
        # estimate conditional copulas
        cop_pp1=edf.eval_cdf(c_cdf_y_pos,qy_pos,y)
        cop_pp2=edf.eval_cdf(c_cdf_x_pos,qx_pos,x)
        cop_pp=np.hstack((cop_pp1,cop_pp2))
        
        cop_np1=edf.eval_cdf_neg(c_cdf_y_neg,qy_neg,y)
        # cop_np1=edf.eval_cdf(c_cdf_y_neg,qy_neg,y-1)
        cop_np2=edf.eval_cdf(c_cdf_x_pos,qx_pos,x)
        cop_np=np.hstack((cop_np1,cop_np2))
        
        cop_pn1=edf.eval_cdf(c_cdf_y_pos,qy_pos,y)
        cop_pn2=edf.eval_cdf_neg(c_cdf_x_neg,qx_neg,x)
        # cop_pn2=edf.eval_cdf(c_cdf_x_neg,qx_neg,x-1)
        cop_pn=np.hstack((cop_pn1,cop_pn2)) 
        
        cop_nn1=edf.eval_cdf_neg(c_cdf_y_neg,qy_neg,y)
        # cop_nn1=edf.eval_cdf(c_cdf_y_neg,qy_neg,y-1)
        cop_nn2=edf.eval_cdf_neg(c_cdf_x_neg,qx_neg,x)
        # cop_nn2=edf.eval_cdf(c_cdf_x_neg,qx_neg,x-1)
        cop_nn=np.hstack((cop_nn1,cop_nn2))
        
        # estimate empirical conditional copula CDFs
        qcop_pp1,qcop_pp2,cop_pp_cdf=edf.ecdf_2D_disc(cop_pp)
        qcop_np1,qcop_np2,cop_np_cdf=edf.ecdf_2D_disc(cop_np)
        qcop_pn1,qcop_pn2,cop_pn_cdf=edf.ecdf_2D_disc(cop_pn)
        qcop_nn1,qcop_nn2,cop_nn_cdf=edf.ecdf_2D_disc(cop_nn)
        
        y_pmf=np.unique(y,return_counts=True)[1]/len(y)
        
        # calculate discrete copula differences
        cd_cdf_pos= ((cop_pp_cdf-cop_np_cdf))/y_pmf[:,np.newaxis]
        cd_cdf_neg= ((cop_pn_cdf-cop_nn_cdf))/y_pmf[:,np.newaxis]
        if len(cd_cdf_pos)>len(cd_cdf_neg):
            cd_cdf_neg=np.hstack((cd_cdf_neg,cd_cdf_neg[-1]))
            if len(cd_cdf_pos)>len(cd_cdf_neg):
                cd_cdf_neg=np.hstack((cd_cdf_neg,cd_cdf_neg[-1]))    
        # find the conditional pmf
        # cd_pmf=abs(sum(cd_cdf_pos-cd_cdf_neg))
        cd_pmf= abs(cd_cdf_pos-cd_cdf_neg)
        cd_pmf= cd_pmf[:-1,:-1]
        cd_pmf_samps_rej=np.zeros((x.size,cd_pmf.shape[0]))
        
        for i in range(cd_pmf.shape[0]):
            
            cd_pmf_temp=cd_pmf[i,:]/sum(cd_pmf[i,:])
            cd_pmf_temp[cd_pmf_temp==0]=1e-8
            x_pmf=np.unique(x)
            if not any(np.isnan(cd_pmf_temp)): 
                cd_pmf_samps_rej[:,i]= rej_sample(cd_pmf_temp,x_pmf,x.size)
            else:
                cd_pmf_samps_rej[:,i]= np.asarray([np.nan])

        cd_pmf_samps_rej=abs(cd_cdf_pos-cd_cdf_neg)
        
    if not x_continuous and y_continuous:
        
        # estimate the different conditional cdfs
        qx_pos,c_cdf_x_pos=edf.ecdf(x)
        qx_neg,c_cdf_x_neg=qx_pos,np.hstack((np.zeros(1,),c_cdf_x_pos[:-1]))
        
        qy , c_cdf_y =edf.ecdf(y)
        
        # estimate conditional copulas
        cop_y=edf.eval_cdf(c_cdf_y,qy,y)
        
        cop_pp2=edf.eval_cdf(c_cdf_x_pos,qx_pos,x)
        cop_cp=np.hstack((cop_y,cop_pp2))
        
        cop_n2=edf.eval_cdf_neg(c_cdf_x_neg,qx_neg,x)
        cop_cn=np.hstack((cop_y,cop_n2))

        # estimate empirical conditional copula CDFs
        qcp1,qcp2,cop_cp_cdf=edf.ecdf_2D_mixed(cop_cp,cont=0,disc=1)
        qcn1,qcn2,cop_cn_cdf=edf.ecdf_2D_mixed(cop_cn,cont=0,disc=1)
        
        # differentiate copula cdf
        # calculate discrete copula differences
        cd_cdf_pos= np.diff(cop_cp_cdf,axis=0)
        cd_cdf_neg= np.diff(cop_cn_cdf,axis=0)
        
        
        if len(cd_cdf_pos)>len(cd_cdf_neg):
            cd_cdf_neg=np.hstack((cd_cdf_neg,cd_cdf_neg[-1]))
            if len(cd_cdf_pos)>len(cd_cdf_neg):
                cd_cdf_neg=np.hstack((cd_cdf_neg,cd_cdf_neg[-1]))    
        # find the conditional pmf
        cd_pmf= abs(cd_cdf_pos-cd_cdf_neg)

        # cd_pmf= cd_pmf[:-1,:-1]
        # cd_pmf_samps_rej=np.zeros((x.size,cd_pmf.shape[0]))
        
        # for i in range(cd_pmf.shape[0]):
            
        #     cd_pmf_temp=cd_pmf[i,:]/sum(cd_pmf[i,:])
        #     cd_pmf_temp[cd_pmf_temp==0]=1e-8
        #     x_pmf=np.unique(x)
        #     if not any(np.isnan(cd_pmf_temp)): 
        #         cd_pmf_samps_rej[:,i]= rej_sample(cd_pmf_temp,x_pmf,x.size)
        #     else:
        #         cd_pmf_samps_rej[:,i]= np.asarray([np.nan])

        cd_pmf_samps_rej=abs(cd_cdf_pos-cd_cdf_neg)
    elif x_continuous and y_continuous:

        # estimate the different conditional cdfs
        qx , c_cdf_x =edf.ecdf(x)
        qy , c_cdf_y =edf.ecdf(y)
        
        # estimate conditional copulas
        cop_1=edf.eval_cdf(c_cdf_x,qx,x)
        cop_2=edf.eval_cdf(c_cdf_y,qy,y)
        cop=np.hstack((cop_1,cop_2))
        
        # estimate empirical conditional copula CDFs
        qcop_1,qcop_2,cop_cdf=edf.ecdf_2D(cop)
        
        # differentiate copula cdf
        cop_diff=np.diff(cop_cdf,axis=0)
        
        
        cd_pmf= np.sum(np.diff(cop_diff),axis=0)
        cd_pmf=cd_pmf/np.sum(cd_pmf)
        x_pmf= qcop_1
        
        # cd_pmf_samps_rej= rej_sample(cd_pmf,x_pmf[:-2],x.size)
        cd_pmf_samps_rej=np.diff(cop_diff)
    # sample from the conditional pmf for flow training
    
    # cd_pmf_samps_inv= inv_sample(cd_pmf,x_pmf,cop_pp2)
    # plt.figure()
    # plt.plot(x_pmf,cd_pmf)
    # plt.hist(cd_pmf_samps_rej,bins=20,alpha=0.5, label='rej', density= 'True')
    # plt.hist(cd_pmf_samps_inv,bins=20,alpha=0.5, label='inv', density= 'True')
    # plt.legend()
    # plt.show()
    return cd_pmf_samps_rej.astype('float64') 

def c_pmf_cond(x,y):
    '''

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    cd_pdf : TYPE
        DESCRIPTION.

    '''
        
    # estimate the different conditional cdfs
    qx_pos,c_cdf_x_pos=edf.ecdf(x)
    qx_neg,c_cdf_x_neg=qx_pos,np.hstack((np.zeros(1,),c_cdf_x_pos[:-1]))
    
    # continuous case
    qy , c_cdf_y =edf.ecdf(y)
    
    # estimate conditional copulas
    cop_p_disc=edf.eval_cdf(c_cdf_x_pos,qx_pos,x)
    cop_n_disc=edf.eval_cdf_neg(c_cdf_x_neg,qx_neg,x)
    cop_cont=edf.eval_cdf(c_cdf_y,qy,y)
    
    cop_cp=np.hstack((cop_p_disc,cop_cont))
    cop_cn=np.hstack((cop_n_disc,cop_cont))
    
    
    # estimate empirical conditional copula CDFs
    qcop_cp1,qcop_cp2,cop_cp_cdf=edf.ecdf_2D_mixed(cop_cp,1,0)
    qcop_cn1,qcop_cn2,cop_cn_cdf=edf.ecdf_2D_mixed(cop_cn,1,0)
    
    
    cop_diff_cp=np.diff(cop_cp_cdf,axis=0)
    cop_diff_cn=np.diff(cop_cn_cdf,axis=0)
    
    # calculate discrete copula differences
    cd_pdf=abs(np.sum(cop_diff_cp-cop_diff_cn,axis=1))
    cd_pdf=cd_pdf/np.sum(cd_pdf)
    x_pdf=np.unique(x)
    if len(x_pdf)-len(cd_pdf)==2:
        cd_pmf_samps_rej=rej_sample(cd_pdf,x_pdf[:-1],x.size)
    else:
        cd_pmf_samps_rej=rej_sample(cd_pdf,x_pdf[:-2],x.size)
    
    return cd_pmf_samps_rej.astype('float64')      