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
    
    num_train = 10000
    batch_train = args.batch_size
    uterus_train_data = uterus(num_train)
    train_data_loader = uterus_train_data.return_dataloader(batch_train)
    
    num_val = 3000
    batch_val = args.batch_size_test
    uterus_val_data = uterus(num_val)
    val_data_loader = uterus_val_data.return_dataloader(batch_val)
    
    uterus_val_data.plot_both()


if __name__ == '__main__':
    main()