# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 03:28:25 2020

@author: lazar
"""

import numpy as np
import numpy.matlib as npmat
import edistr_funcs as edf
from scipy.stats import randint

def inv_sample(pdf,x_pdf,unif_samps):

    emp_cdf=np.cumsum(pdf)/np.sum(pdf)


    inv_samps=edf.eval_inv_cdf(x_pdf,emp_cdf,unif_samps)
    return inv_samps

def inv_sample_nd(pdf,x_pdf,n_samps):
    
    dims = len(pdf.shape)
    emp_cdf=np.empty((dims), dtype=object)
    marg=np.empty((dims), dtype=object)
    idx_sum=[1,0]
    for i in range(dims):
        marg[i]=np.sum(pdf,axis=idx_sum[i])/np.sum(np.sum(pdf,axis=idx_sum[i]))
        emp_cdf[i]=np.cumsum(marg[i])/np.sum(marg[i])

    
    unif_samps=np.random.uniform(0,1,(dims,n_samps))
    inv_samps=np.zeros((dims,n_samps))
    for i in range(dims):
        inv_samps[i,:]=edf.eval_inv_cdf(x_pdf[i],emp_cdf[i],unif_samps[i,:])
    return inv_samps

def rej_sample_2d(pdf,x_pdf,n_samp):
    # rejection sampling for drawing samples from the estimated density
    out= np.zeros((n_samp,2))
    init_samps=100000
    x_i=np.zeros((init_samps,2))
    pdf_idx=np.zeros((init_samps,2))
    
    for i in range(2): 
        if np.round(x_pdf[i][1])==x_pdf[i][1]: # integer
            x_i[:,i]=randint.rvs(min(x_pdf[i]),max(x_pdf[i]),size=init_samps)
            x_mat=npmat.repmat(x_pdf[i],init_samps,1)
            difs=abs(npmat.repmat(x_i[:,i],len(x_pdf[i]),1).T-x_mat)
            pdf_idx[:,i]=np.argmin(difs,axis=1)
        else:
            x_i[:,i]=np.random.uniform(min(x_pdf[i]),max(x_pdf[i]),(init_samps,))
            x_mat=npmat.repmat(x_pdf[i],init_samps,1)
            difs=abs(npmat.repmat(x_i[:,i],len(x_pdf[i]),1).T-x_mat)
            pdf_idx[:,i]=np.argmin(difs,axis=1)
    
    pdf_idx=pdf_idx.astype(int)
    y_i=np.random.uniform(0,np.max(np.max(pdf)),init_samps)
    px_i=pdf[pdf_idx[:,0],pdf_idx[:,1]]
    
    out1=x_i[y_i<px_i,0]
    out2=x_i[y_i<px_i,1]
    if any(out1<x_pdf[0][0]) or any(out1>x_pdf[0][-1]):
        out_temp=out1
        out1=np.delete(out1,out_temp<x_pdf[0][0])
        out1=np.delete(out1,out_temp>x_pdf[0][-1])
        out2=np.delete(out2,out_temp<x_pdf[0][0])
        out2=np.delete(out2,out_temp>x_pdf[0][-1])
    if any(out2<x_pdf[1][0]) or any(out2>x_pdf[1][-1]):
        out_temp=out2
        out1=np.delete(out1,out_temp<x_pdf[0][0])
        out1=np.delete(out1,out_temp>x_pdf[0][-1])
        out2=np.delete(out2,out_temp<x_pdf[0][0])
        out2=np.delete(out2,out_temp>x_pdf[0][-1])
    
    out=np.vstack((out1[:n_samp],out2[:n_samp]))
    
    return out

def rej_sample(pdf,x_pdf,n_samp):
    # rejection sampling for drawing samples from the estimated density
    out= np.zeros((n_samp,1))
    init_samps=10000
    x_i=np.zeros((init_samps,1))
    pdf_idx=np.zeros((init_samps,1))
    
    if np.round(x_pdf[1])==x_pdf[1]: # integer
        x_i=randint.rvs(min(x_pdf),max(x_pdf),size=init_samps)
        x_mat=np.matlib.repmat(x_pdf,init_samps,1)
        difs=abs(np.matlib.repmat(x_i,len(x_pdf),1).T-x_mat)
        pdf_idx=np.argmin(difs,axis=1)
    else:
        x_i=np.random.uniform(min(x_pdf),max(x_pdf),(init_samps,))
        x_mat=np.matlib.repmat(x_pdf,init_samps,1)
        difs=abs(np.matlib.repmat(x_i,len(x_pdf),1).T-x_mat)
        pdf_idx=np.argmin(difs,axis=1)
    
    pdf_idx=pdf_idx.astype(int)
    if np.max(pdf_idx)>len(pdf)-1:
        pdf_idx[pdf_idx==max(pdf_idx)]=max(pdf_idx)-1
    y_i=np.random.uniform(0,np.max(np.max(pdf)),init_samps)
    px_i=pdf[pdf_idx]
    
    out=x_i[y_i<px_i]

    rd_idx=np.random.randint(0,high=len(out)-1,size=n_samp)
    out=out[rd_idx]
    
    return out