# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 22:23:04 2020

@author: lazar
"""
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

from nflows.flows.base import Flow
from nflows.distributions.uniform import BoxUniform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.autoregressive import \
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform

import edistr_funcs as edf

# device='cpu'
device='cuda'
class ns_flow():
    
    '''
    this module implements a Neural Spline Flow model (NSF) with a uniform 
    base distribution so as to approximate the probability transform to map 
    samples to and from copula space with uniform margins
    
    '''
    
    
    def __init__(self,n_features,n_layers,n_units,dropout_pr,
                 is_continuous,n_bins=4,res_blocks=2):
        '''

        Parameters
        ----------
        n_features : int, corresponds to the number of dimensions of the samples
        n_layers : int, number of layers in the NSF model
        n_units : int, numer of hidden units in the NSF model
        is_continuous : bool, True for continuous data, False for discrete
        dropout_pr: float, [0,1], dropout probability in the NSF model
        n_bins: int, corresponds to the number of knots of the splines
        res_blocks: int, number of residual blocks in the NSF model


        '''
        self.n_features=n_features
        self.n_layers=n_layers
        self.n_units=n_units
        self.dropout_pr=dropout_pr
        self.n_bins=n_bins
        self.res_blocks=res_blocks
        self.is_continuous=is_continuous
        
    def train(self,samples):
        
        '''
        fits the NSF model to data
        
        Parameters
        ----------
        samples : numpy array, input data dimensions-by-N_samples   
    
        Returns
        -------
        flow: trained instance of the flow_vine class
    
        '''
        
        num_layers =self.n_layers
        if len(samples.shape)==1:
            l=Variable(torch.from_numpy(-0.01**np.ones((self.n_features,)))).to(device)
        else:
            l=Variable(torch.from_numpy(-0.01**np.ones((self.n_features,)))).to(device)
        h=Variable(torch.from_numpy(1.01*np.ones((self.n_features,)))).to(device)
        base_dist = BoxUniform(low=l,high=h).to(device)
        
        n_samp=samples.shape[0]
        
        transforms = []
        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=self.n_features))
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.n_features, hidden_features=self.n_units,
                context_features=2, num_blocks=self.res_blocks,num_bins=self.n_bins,
                tails='linear',tail_bound=np.round(np.max(samples))+0.01,
                dropout_probability=self.dropout_pr))
            # transforms.append(MaskedPiecewiseLinearAutoregressiveTransform(
            #     features=self.n_features, hidden_features=self.n_units,
            #     context_features=2, num_blocks=self.res_blocks,num_bins=self.n_bins,
            #     dropout_probability=self.dropout_pr))
        transform = CompositeTransform(transforms)
        
        # transforms = []
        # for _ in range(num_layers):
        #     transforms.append(ReversePermutation(features=self.n_features))
        #     transforms.append(MaskedAffineAutoregressiveTransform(
        #         features=self.n_features, hidden_features=self.n_units,
        #         context_features=2))
        # transform = CompositeTransform(transforms)


        num_iter = 1000
        
        self.flow = Flow(transform, base_dist).to(device)
        optimizer = optim.Adam(self.flow.parameters())
        
        # normalize samples
        if len(samples.shape)==1:
            x = torch.tensor(np.reshape(samples,(n_samp,self.n_features)),
                             dtype=torch.float32,device=device)
        else:
            x = torch.tensor(samples, dtype=torch.float32,device=device)
        
        for i in range(num_iter):
            optimizer.zero_grad()
            loss = -self.flow.log_prob(inputs=x).mean()
            loss.backward()
            optimizer.step()
            # if i % 400 ==0:
            #     print(str(i))
            #     print(str(loss))
        
        return self.flow
    
    
    def sample(self,data,inputs_base):
        
        '''
        draws simulated data from a trained NSF model by transforming 
        samples from a uniform distribution
        
        Parameters
        ----------
        data : numpy array, the samples used for training. Used for rescaling 
               NSF samples 
        inputs base: numpy array, samples from a uniform distribution
    
        Returns
        -------
        flow_output: samples transformed by the NSF model
    
        '''
        
        # bring to scale 
        if torch.is_tensor(data):
            inputs_base = inputs_base*np.max(data.cpu().detach().numpy())
        else:
            inputs_base = inputs_base*np.max(data)
        
        with torch.no_grad():
            inputs_base= torch.tensor((inputs_base).astype(np.float32),device=device)
            z = self.flow.inv_transform(inputs_base)[0]
        
        if self.is_continuous==True:
            flow_output=z.cpu().detach().numpy()
        else:
            flow_output=np.round(z.cpu().detach().numpy()) 
            
        return flow_output
    
    def transform_to_base(self,data,cond_cop=False):
        
        '''
        transforms data to a uniform base distribution 
        
        Parameters
        ----------
        data : numpy array, the samples used for training. Used for rescaling 
               NSF samples 
        cond_cop: to be removed
    
        Returns
        -------
        base: samples transformed to base uniform by the NSF model
        x_base: to be removed
        
        '''
        
        n_samp=data.shape[0]
        data=torch.tensor(
            np.reshape(data,(int(n_samp),self.n_features)), dtype=torch.float32,
            device=device)
        # base=self.flow.transform_to_noise(data)
        
        # bring to scale
        data=data.cpu().detach().numpy()
        # base=base.detach().numpy()
        # base=base/max(data)
        # base[base>max(data)]=max(data)-0.001
        x_base, base_cdf= edf.ecdf(data)
        base= edf.eval_cdf(base_cdf,x_base,data)

        return base, x_base
    
    def density(self,points):
        
        '''
        estimates NSF density given a grid
        
        Parameters
        ----------
        points: grid
    
        Returns
        -------
        dens: NSF density on the grid provided
        
        
        '''
        
        # size= grid.shape[1]
        points=torch.tensor(points.astype(np.float32),device=device).T
        # y_grid=torch.tensor((grid[1,:]).astype(np.float32))
        # xgrid, ygrid = torch.meshgrid(x_grid, y_grid)
        # xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
        z_dens= self.flow.log_prob(points)
        dens= np.exp(z_dens.cpu().detach().numpy())
        return dens