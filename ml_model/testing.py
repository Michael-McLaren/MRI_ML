#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 19:19:27 2022

@author: s1740929
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_generation import uterus
from arg_extractor import get_args
from experiment_build import ExperimentBuilder
from model_architectures import BasicNet
import os 

#python ml_model/testing.py --num_epochs 100 --experiment_name trained_models/testing


def main():
    
    args = get_args()  # get arguments from command line
    #rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
    #torch.manual_seed(seed=args.seed)  # sets pytorch's seed
    
    # set up data augmentation transforms for training and testing

    #figure out how to do this
    #transform_test = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #])
    name = args.experiment_name.split('/')
    name = name[-1]
    
    num_train = 5
    batch_train = args.batch_size
    uterus_train_data = uterus(num_train)
    
    #LOAD DATA
    uterus_train_data.load_data('trained_models', name+'_train_data')
    train_data_loader = uterus_train_data.return_dataloader(batch_train, shuffle = False)
    
    num_val = 5
    batch_val = args.batch_size_test
    uterus_val_data = uterus(num_val)
    
    #LOAD DATA
    uterus_val_data.load_data('trained_models', name + '_val_data')
    val_data_loader = uterus_val_data.return_dataloader(batch_val, shuffle = False)
    
    custom_conv_net = BasicNet()

    mri_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        weight_decay_coefficient=args.weight_decay_coefficient,
                                        pk_weight = args.pk_weight,
                                        lr = args.lr,
                                        curve_weight = args.curve_weight,
                                        train_data=train_data_loader, 
                                        val_data=val_data_loader,
                                        test_data=None)  # build an experiment object
    

    
    model_path = '/afs/inf.ed.ac.uk/user/s17/s1740929/MRI_ML/trained_models/'+name+'/saved_models'
    mri_experiment.load_model(model_path, 'train_model', 49)
    
    
    x = np.load('trained_models/data/test_x.npy')
    y = np.load('trained_models/data/test_y.npy')
    
    j = 59
    x_norm = uterus_train_data.normalise(x)
    mri_experiment.example_fit(x[j], y[j], x_norm[j])
    
    #train data
    x = uterus_train_data.x
    y = uterus_train_data.y
    
    j = 2
    x_norm = uterus_train_data.normalise(x)
    mri_experiment.example_fit(x[j], y[j], x_norm[j])
    

if __name__ == '__main__':
    main()