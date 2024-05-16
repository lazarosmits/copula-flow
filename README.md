# Non-parametric C-Vine copula density estimation with Neural Spline Flows (NSF)

This repository contains code that was used in the paper [Mixed vine copula flows for flexible modeling of neural dependencies](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.910122/full) 

Given continuous or discrete joint observations, it builds a C-Vine and fits NSF models for the margins and the pair copulas of the joint distribution. Doing so, it wraps normalizing flow-based density estimators from [nflows](https://github.com/bayesiains/nflows)


...
import numpy as np
import flow_vine as flow_vine
import matplotlib.pyplot as plt

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
n_samp=10000
samples = vine.rvs(n_samp).T
samples=samples+np.abs(np.min(samples)) # bring to positive values for NSF fitting

# visualize samples per each pair to inspect joint distributions 
plt.figure()
plt.rc('font',size=20)
cop_idx= np.identity(dim-1)
cop_idx[np.triu_indices(dim-1)]=1
icount=1
cop_count=0
ylab_count=0
for i in range(dim-1):
    for j in range(dim-1):
        if np.any(cop_idx[i,j]==1):
            plt.subplot(dim-1,dim-1,icount)
            plt.scatter(samples[i,:],samples[j+1,:],
                        s=2,alpha=0.1)
            plt.xticks([])
            plt.yticks([])
            cop_count+=1
        icount+=1
...
