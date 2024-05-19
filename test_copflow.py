# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:55:19 2024

@author: lazar
"""

import numpy as np
import flow_vine as flow_vine
import matplotlib.pyplot as plt
import seaborn as sns

from mixedvines.copula import GaussianCopula, ClaytonCopula
from mixedvines.mixedvine import MixedVine
from scipy.stats import norm, gamma, poisson

# Manually construct 5-D C-vine
dim = 5  # Dimension
vine = MixedVine(dim)

# Specify marginals with different distributions
vine.set_marginal(0, norm(4, 2))
vine.set_marginal(1, gamma(2, 3, 3))
vine.set_marginal(2, norm(7, 3))
vine.set_marginal(3, gamma(2, 3, 3))
vine.set_marginal(4, norm(9,4))

# Specify pair copulas
deg=['90°','180°','270°'] # for rotated versions of copulas
dim_range= np.linspace(0,dim-1,dim)

# I am using (rotated) Clayton and Gaussian copulas
while len(dim_range)>0:
    for i in range(len(dim_range)-1):
        if np.any([i==x for x in range(3)]):
            vine.set_copula(int(dim_range[0])+1, i,  ClaytonCopula(3,rotation=deg[i]))
        else:
            vine.set_copula(int(dim_range[0])+1, i,  GaussianCopula(0.7))
    dim_range=np.delete(dim_range,0)


# draw samples from the C-vine
n_samp=15000
samples = vine.rvs(n_samp).T
samples=samples+np.abs(np.min(samples)) # bring to positive values for NSF fitting

# visualize samples per each pair to inspect joint distributions 

# labels of variables for axes in figures
xlabels=['2','3','4','5','3|1','4|1','5|1',
         '4|1,2','5|1,2','5|1,2,3']
ylabels=['1','2|1','3|1,2','4|1,2,3']

plt.figure()
plt.rc('font',size=12)
# create an index to give subplots an upper triangular matrix arrangement
cop_idx= np.identity(dim-1)
cop_idx[np.triu_indices(dim-1)]=1
# index to count the subplots
icount=1
# index to track labels of y axis
ylab_count=0
# index to track labels of x axis
xlab_count=0
for i in range(dim-1):
    for j in range(dim-1):
        if np.any(cop_idx[i,j]==1):
            plt.subplot(dim-1,dim-1,icount)
            plt.scatter(samples[j+1,:],samples[i,:],
                        s=2,alpha=0.05)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(xlabels[xlab_count])
            if i==j: # only label y axis of leftmost panel
                plt.ylabel(ylabels[ylab_count])
                ylab_count+=1
            xlab_count+=1
        icount+=1
plt.suptitle('Joint distributions per pair of variables')

# plot an example pair of variables with margins on the sides
pair_data=np.vstack((samples[1,:],samples[0,:]))

g = sns.JointGrid()
sns.scatterplot(x=samples[1,:],y=samples[0,:],
                linewidth=0.5,s=15,ax=g.ax_joint)
g.ax_joint.set_xlabel('var 2')
g.ax_joint.set_ylabel('var 1')

sns.histplot(x=samples[1,:], fill=True,ec='k', linewidth=0.5, bins=50, ax=g.ax_marg_x)
sns.histplot(y=samples[0,:], fill=True,ec='k', linewidth=0.5, bins=50, ax=g.ax_marg_y)
#%% Fit a 5-D C-vine 

dim=samples.shape[0]

vine_5d = flow_vine.flow_mixed_vine(n_dims=dim)
(emp_copulas, copulas, cop_densities,
  marginals, r_margins, var_order) = vine_5d.build_Cvine(
      samples,is_continuous=True)
      
#%% visualize margins and copulas




# plot histograms of the flow margins s the real margins
plt.figure()
plt.rc('font',size=12)
for i in range(dim):
    plt.subplot(1,5,i+1)
    plt.hist(samples[var_order[i],:],bins=50,color='b',alpha=0.5,label='real')
    plt.hist(marginals[i],bins=50,color='r',alpha=0.5,label='flow')
    plt.title('Margin '+str(i+1))
    plt.xlabel('value')
    if i==0:
        plt.ylabel('count')
        plt.legend()


# plot empirical copulas
plt.figure()
plt.rc('font',size=14)
# index to arrange the subplots like an upper triangular matrix
cop_idx= np.identity(dim-1)
cop_idx[np.triu_indices(dim-1)]=1
# index to position the subplots in the right place
icount=1
# index to keep track of the copulas being plotted
cop_count=0
# index to track labels of y axis
ylab_count=0
for i in range(dim-1):
    for j in range(dim-1):
        if np.any(cop_idx[i,j]==1):
            plt.subplot(dim-1,dim-1,icount)
            plt.scatter(emp_copulas[cop_count][:,0],emp_copulas[cop_count][:,1],
                        s=2,alpha=0.02)
            plt.xlabel(xlabels[cop_count])
            if i==j: # only label y axis of leftmost panel
                plt.ylabel(ylabels[ylab_count])
                ylab_count+=1
            plt.xticks([])
            plt.yticks([])
            cop_count+=1
        icount+=1
plt.show()


# plot copula densities
plt.figure()
plt.rc('font',size=14)
cop_idx= np.identity(dim-1)
cop_idx[np.triu_indices(dim-1)]=1
icount=1
cop_count=0
ylab_count=0
xx=np.linspace(0,1,200)
for i in range(dim-1):
    for j in range(dim-1):
        if np.any(cop_idx[i,j]==1):
            plt.subplot(dim-1,dim-1,icount)
            if cop_count==len(copulas)-1:
                plt.pcolor(xx,xx,cop_densities[cop_count].reshape(len(xx),len(xx)),
                           shading='auto')
                plt.xlabel(xlabels[cop_count])
                if i==j:
                    plt.ylabel(ylabels[ylab_count])
                    ylab_count+=1
            else:
                plt.pcolor(xx,xx,cop_densities[cop_count].reshape(len(xx),len(xx)),
                           shading='auto')
                plt.xlabel(xlabels[cop_count])
                if i==j:
                    plt.ylabel(ylabels[ylab_count])
                    ylab_count+=1
            cop_count+=1
            plt.yticks([])
            plt.xticks([])
        icount+=1
plt.suptitle('Flow Copula densities')
plt.show()