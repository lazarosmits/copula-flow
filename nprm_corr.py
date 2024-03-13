# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:11:03 2020

@author: lazar
"""

import numpy as np
from scipy.stats import kendalltau

def kendall_matrix(data):
    
    dims=data.shape[0]
    corr_mat=np.zeros((dims,dims))
    i=0
    
    while i<=dims:
        for j in range(i,dims):
            corr_mat[i,j] = kendalltau(data[i,:],data[j,:])[0]
            corr_mat[j,i] = corr_mat[i,j]
        i+=1
        
    return corr_mat