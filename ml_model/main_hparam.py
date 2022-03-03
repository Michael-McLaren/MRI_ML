#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:11:42 2022

@author: s1740929
"""


import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_generation import uterus
from arg_extractor import get_args
from experiment_build import ExperimentBuilder
from model_architectures import BasicNet, NeuralNet, NeuralNet_Dropout
import os 


def main():
    args = get_args()  
    
    args = get_args()  
    
    batch_train = args.batch_size

    batch_val = args.batch_size_test
    
    num = 100000
    
    uterus_data = uterus(num, smooth = True)
    train, val, test, scaler = uterus_data.create_synth_data(batch_train, batch_val, shuffle = True)
    
    hyper_dict = {}
    length = 100
    for i in range(length):
        enc_length = np.random.randint(high = 10, size =1)
        enc_sizes = list(np.random.randint(low = 50, high = 500, size = enc_length))
        p1 = 0.5 + np.random.random(size =1)/2
        hyper_dict[i] = (enc_sizes, p1)
    
    metrics_dict = {}
    for i, value in hyper_dict.items():
        enc_sizes, p1 = value #unpack
        

        custom_conv_net = NeuralNet_Dropout(enc_sizes, p = p1)
        model_str = str(custom_conv_net)
        
        mri_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                            experiment_name=args.experiment_name,
                                            num_epochs=args.num_epochs,
                                            weight_decay_coefficient=args.weight_decay_coefficient,
                                            pk_weight = args.pk_weight, 
                                            curve_weight = args.curve_weight,
                                            lr = args.lr,
                                            train_data=train, 
                                            val_data=val,
                                            test_data=test, 
                                            scaler = scaler,
                                            save = False)  # build an experiment object
        
    
        
        best_val_model_loss, best_val_model_idx, best_val_model_loss_curve = mri_experiment.run_experiment()  # run experiment and return experiment metrics
        print('\n best val model loss: ', best_val_model_loss)
        print('\n best val model idx: ', best_val_model_idx)
        print('\n best val model loss curve: ', best_val_model_loss_curve)
    
        
        
        #larger is better
        stats, res = mri_experiment.pk_dist()
        
        #store test metrics
        metrics_dict[i] = (model_str,
                    best_val_model_loss,
                    best_val_model_loss_curve,
                    best_val_model_idx,
                    res[0], 
                    res[1], 
                    res[2],
                    stats[0], 
                    stats[1], 
                    stats[2])
    
    
if __name__ == '__main__':
    main()