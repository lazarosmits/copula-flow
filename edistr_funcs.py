# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 23:34:02 2020

@author: lazar
"""
import numpy as np
from scipy.stats import uniform,hypergeom,norm

def ecdf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def ecdf_neg(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def ecdf_2D(sample,bins=32):

    # find the unique values and their corresponding counts
    epdf,qtl1,qtl2= np.histogram2d(sample[:,0],sample[:,1],bins)
    
    cumprob = np.cumsum(np.cumsum(epdf,axis=0),axis=1)/sample.shape[0]
    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1

    return qtl1, qtl2,cumprob

def ecdf_2D_disc(sample):

    # find the unique values and their corresponding counts
    pmf1= np.unique(sample[:,0])
    pmf2= np.unique(sample[:,1])
    cumprob = np.zeros((len(pmf1),len(pmf2)))
    for i in range(len(pmf1)):
        for j in range(len(pmf2)):
            cumprob[i,j]=len(sample[np.logical_and(sample[:,0]<=pmf1[i],
                                        sample[:,1]<=pmf2[j]),0])
    cumprob= cumprob/sample.shape[0]
    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1

    return pmf1, pmf2,cumprob

def ecdf_2D_mixed(sample,cont,disc):

    # find the unique values and their corresponding counts
    pmf_un= np.unique(sample[:,disc])
    
    epdf,qtl_disc,qtl_cont = np.histogram2d(sample[:,disc],sample[:,cont],
                             bins=[len(pmf_un)-1,32])
    cumprob = np.cumsum(np.cumsum(epdf,axis=0),axis=1)/sample.shape[0]
    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1

    return qtl_disc, qtl_cont,cumprob.T

def eval_cdf_2D(quant1,quant2,prob,samples):
    qq1,qq2=np.meshgrid(quant1,quant2)
    qq_points=np.array([qq1.ravel(), qq2.ravel()]).T
    cdf_unrvl=prob.ravel()
    int_transf=np.zeros((samples.shape[0],1))
    for i in range(samples.shape[0]):
      #how to use 2d cdf for integral transform?
       difs1=abs(samples[i,0]-quant1)
       idx1=np.where(difs1==min(difs1))
       difs2=abs(samples[i,1]-quant2)
       idx2=np.where(difs2==min(difs2))
       pr_idx=np.where((qq_points[:,0]==
           quant1[int(idx1[0])]) & (qq_points[:,1]== quant2[int(idx2[0])]))[0]
       if pr_idx.size<1:
           pr_idx=pr_idx[0,:]
       int_transf[i]=cdf_unrvl[int(pr_idx)]
       # print('sample'+str(samples[i,0])+', '+str(samples[i,1])+': '+str(int_transf[i]))
    # int_transf=np.reshape(int_transf,(len(qq1),len(qq2)))
    return int_transf

def eval_cdf(cdf,x_cdf,x_samps):
    
    transf=np.zeros((x_samps.size,1))
    for i in range(x_samps.shape[0]):
       difs=abs(x_samps[i]-x_cdf)
       idx=np.where(difs==min(difs))[0]
       # print(str(min(difs)))
       if idx.size<1:
           print(str(x_samps[i]))
       if idx.size>1:
           transf[i]=np.mean([cdf[int(idx[0])],
                             cdf[int(idx[1])]])
       else:
           transf[i]=cdf[int(idx)]
               
       # print('sample: '+str(x_samps[i])+', P: '+str(transf[i]))
       
    return transf

def eval_cdf_neg(cdf,x_cdf,x_samps):
    
    transf=np.zeros((x_samps.size,1))
    for i in range(x_samps.shape[0]):
       difs=abs(x_samps[i]-x_cdf)
       idx=np.where(difs==min(difs))[0]-1
       # print(str(min(difs)))
       if idx.size<1:
           print(str(x_samps[i]))
       if idx.size>1:
           transf[i]=np.mean([cdf[int(idx[0])],
                             cdf[int(idx[1])]])
       else:
           transf[i]=cdf[int(idx)]
               
       # print('sample: '+str(x_samps[i])+', P: '+str(transf[i]))
       
    return transf

def eval_inv_cdf(inv_cdf,x_icdf,x_samps):
    
    inv_transf=np.zeros((x_samps.size,))
    for i in range(x_samps.shape[0]):
       difs=abs(x_samps[i]-x_icdf)
       idx=np.where(difs==min(difs))
       if idx[0].size>1:
           inv_transf[i]=inv_cdf[int(np.random.choice(idx[0]))]
       else:
           inv_transf[i]=inv_cdf[int(idx[0])]
         
    return inv_transf 

def distr_tf(flow_input,flow_output):
    # convert sample to a numpy array, if it isn't already
    flow_input = np.atleast_1d(flow_input)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(flow_input, return_counts=True)
    # counts=np.sort(samples)
    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    F_x=np.zeros((len(quantiles),))
    F_xm=np.zeros((len(quantiles),))
    out=np.zeros((len(flow_output),1))
    # qq= np.zeros((1,))
    for i in range(len(quantiles)):
        F_xm[i] = sum(flow_input<quantiles[i]) / flow_input.size
        F_x[i] = sum(flow_input<=quantiles[i]) / flow_input.size
        # v_temp=[np.random.uniform(F_xm[i],F_x[i],
        #                size=sum(flow_input<=quantiles[i])-sum(flow_input<quantiles[i]))]
        # out=np.append(out,v_temp)
        # qq=np.append(qq,quantiles[i]*np.ones((counts[i])))
    
    for i in range(len(flow_output)):
        # find its range and assign a random number from uniform distribution
        if flow_output[i]==0:
            low=0
        else:
            low=flow_output[i]-F_xm
            low=F_xm[low>0]
            low=low[-1]
        if flow_output[i]==1:
            hi=1
        else:
            hi=flow_output[i]-F_x            
            hi=F_x[hi<=0]
            hi=hi[0]
            
        out[i,:]=np.random.uniform(low,hi,size=1)
    return out

# def inv_distr_tf()
# x=hypergeom.rvs(20, 5, 12,size=20)
# y=np.round(np.random.uniform(0,10, size=20))/10
# z=distr_tf(x,y)