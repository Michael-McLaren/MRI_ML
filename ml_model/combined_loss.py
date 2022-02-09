#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:38:01 2022

@author: s1740929
"""
import torch
import numpy as np

from tkmodel.TwoCUM_copy import TwoCUMfittingConc
from tkmodel.TwoCUM_copy import TwoCUM

def TwoCUM_batch(E, Fp ,vp, AIF1 , t): 
    batch_size = E.shape[0]
    
    Tp=(vp/Fp)*(1-E)
    #Calculate the IRF
    exptTp= torch.exp(-1*t[:,None]/Tp[None,:]) #adding dummy variables so it divides properly

    R=exptTp*(1-E) + E

    #Calculate the convolution
    R = torch.flip(R, (0,)).T #Reshape to fit the the convolution
    R = torch.unsqueeze(R, 1)
    temp = t[1]*torch.nn.functional.conv1d(AIF1, R, padding = AIF1.shape[2]-1).view(batch_size, -1)
    F = Fp.unsqueeze(-1)*temp[:,0:len(t)] #unsqueeze to match dimensions
    return F

def loss_fn_batch(outputs, targets):
    #E, Fp, vp
    #time spacings
    t = np.arange(0,366,2.45)
    t = torch.tensor(t)
    
    batch_size = outputs[:,0].shape[0]
    AIF = torch.from_numpy(np.load("data/AIF.npy")) #WOULD THIS SLOW IT DOWN
    AIF1 = AIF.view(1, 1, -1) #reshaped for convolution

    
    #For outputs
    #First calculate the parameter Tp
    E, Fp ,vp = outputs[:,0], outputs[:,1], outputs[:,2]
    F_out = TwoCUM_batch(E, Fp ,vp, AIF1, t)

    
    #For targets - copy pasted
    E_true, Fp_true ,vp_true = targets[:,0], targets[:,1], targets[:,2]
    F_targets = TwoCUM_batch(E_true, Fp_true ,vp_true, AIF1, t)

    
    MSE = torch.sum((F_out - F_targets)**2)/F_out.shape[1]
    return MSE

def combined(outputs, targets, pk_weight = 50, curve_weight = 1):
    MSE_comb = curve_weight * loss_fn_batch(outputs, targets) + pk_weight*torch.sum((outputs - targets)**2)

    return MSE_comb