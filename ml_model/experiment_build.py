#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 19:39:30 2022

@author: s1740929
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time
import csv


class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data, weight_decay_coefficient):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()
        
        self.experiment_name = experiment_name
        self.model = network_model
        
        #for later use in saving and loading models
        self.state = dict()
        self.starting_epoch = 0

        
        self.model.reset_parameters()  # re-initialize network parameters
        self.train_data = None 
        self.val_data = None
        self.test_data = None
        
        self.optimizer = optim.AdamW(self.parameters(),
                                    weight_decay=weight_decay_coefficient)
        
        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        
        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)  # create the experiment log directory
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        
        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_loss = 0.
        
        self.num_epochs = num_epochs
        self.criterion = combined  # send the loss computation to the GPU
        self.AIF = torch.from_numpy(np.load("../data/AIF.npy"))
        self.time = np.arange(0,366,2.45)
        
    def run_train_iter(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        output = self.model.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
           
        return loss
    
    def run_evaluation_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        with torch.no_grad(): #removes gradient from torch vectors
                self.eval() # sets the system to validation mode for dropout layers and such
                output = self.model.forward(x)
                loss = self.criterion(output, y)
        
        return loss
    
    def save_model(self, model_save_dir, model_save_name, model_idx, best_validation_model_idx,
                   best_validation_model_acc):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.
        """
        self.state['network'] = self.state_dict()  # save network parameter and other variables.
        self.state['best_val_model_idx'] = best_validation_model_idx  # save current best val idx
        self.state['best_val_model_acc'] = best_validation_model_acc  # save current best val acc
        torch.save(self.state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath