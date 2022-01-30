#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:41:40 2022

@author: s1740929
"""
'''This is for the uterus '''
import numpy as np
import scipy.stats as st

from tkmodel.TwoCUM_copy import TwoCUMfittingConc
from tkmodel.TwoCUM_copy import TwoCUM

def E_distribution(num):
    p = 0.09392265193370165
    params = (69.70266807226483, 1.0792465008741048, -0.23357477054649745, 0.08955766184221467)
    start_nums = int(p*num)
    
    start = np.random.uniform(low = 0, high= 0.001, size = start_nums)
    
    dist_num = num - start_nums
    gen_data_Fp = st.exponweib.rvs(params[0], params[1], params[2], params[3], size=dist_num)
    gen_data_Fp = np.array(gen_data_Fp)
    
    true_dist = np.concatenate((start, gen_data_Fp), axis=None)
    
    true_dist[true_dist < 0] = 0.0001

    
    
    return true_dist

def Fp_distribution(num):
    p = 0.1182
    params = (1.064799958220838, 0.0005001496965058328, 0.007151833882619035)
    start_nums = int(p*num)
    
    start = np.random.uniform(low = 0, high= 0.0005, size = start_nums)
    
    dist_num = num - start_nums
    gen_data_Fp = st.weibull_min.rvs(params[0], params[1], params[2], size=dist_num)
    gen_data_Fp = np.array(gen_data_Fp)
    
    true_dist = np.concatenate((gen_data_Fp, start), axis=None)
    #true_dist[true_dist > 1] = 0.99

    
    
    return true_dist

def vp_distrubition(num):
    params = (29572479.030125827, 6450136.935704315, 6450136.715197781)
    p = 0.287
    endnum = int(p*num)
    end = np.random.uniform(low = 0.98, high= 0.99, size = endnum)
    
    dist_num = num - endnum
    gen_data_vp = st.weibull_max.rvs(params[0], params[1], params[2], size=dist_num)
    gen_data_vp = np.array(gen_data_vp)
    
    true_dist = np.concatenate((gen_data_vp, end), axis=None)
    true_dist[true_dist > 1] = 0.99
    true_dist[true_dist < 0] = 0.011

    
    
    return true_dist


def generate_xy(num_curves):
    AIF = np.load("../data/AIF.npy")
    data_size = AIF.shape[0]
    t = np.arange(0,366,2.45)

    #E = np.random.rand(1,num_curves) #0 to 1 for both E and vp
    #vp = np.random.rand(1,num_curves)
    #Fp = np.random.rand(1,num_curves) #this will be multplied by 1e-5 to get the right scale at the end
    
    E = E_distribution(num_curves) 
    Fp = Fp_distribution(num_curves)
    vp = vp_distrubition(num_curves)

    E = E[None,:]
    Fp = Fp[None,:]
    vp = vp[None,:]



    
    E_Fp = np.concatenate((E, Fp), axis =0)
    y = np.concatenate((E_Fp, vp), axis =0)


    x = np.zeros((num_curves, data_size))
    for i in range(num_curves):
        x[i] = TwoCUM(y[:,i], t, AIF, 0)

    y = y.T 
    
    return x,y