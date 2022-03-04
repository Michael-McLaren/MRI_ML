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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.stats import ks_2samp

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from combined_loss import combined, MSE_pk, MSE_curve
from utils import LRScheduler, EarlyStopping
from tkmodel.TwoCUM_copy import TwoCUM

class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data, weight_decay_coefficient, pk_weight, curve_weight, scaler, 
                 lr, save = True, use_gpu = True):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: numpy array
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

        # read more about using moduledict
        #self.model.reset_parameters()  # re-initialize network parameters
        self.train_data = train_data 
        self.val_data = val_data
        self.test_data = test_data
        
        #used to reverse the scale transformation
        self.scaler = scaler
        #used for inverse transformations
        
        self.save = save
        
        self.optimizer = optim.AdamW(self.parameters(),
                                    weight_decay=weight_decay_coefficient,
                                    lr = lr)
        
        self.lr_scheduler = LRScheduler(self.optimizer)
        self.early_stopping = EarlyStopping(patience = 10)
        
        # Generate the directory names
        if self.save:
            self.experiment_folder = os.path.abspath(experiment_name)
            self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
            self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
            
            if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
                os.mkdir(self.experiment_folder)  # create the experiment directory
                os.mkdir(self.experiment_logs)  # create the experiment log directory
                os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory
        
        #send to device
        if use_gpu and torch.cuda.device_count() == 1:
            self.device =  torch.cuda.current_device()
            self.model.to(self.device)  # sends the model from the cpu to the gpu
            print('Use GPU', self.device)
            
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)
            
        # Set best models to be at x since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_loss = 1e10
        self.best_val_model_loss_curve = 1e10
        self.best_val_model_loss_pk = 1e10

        
        self.num_epochs = num_epochs
        self.criterion = combined 
        self.criterion_pk = MSE_pk
        self.criterion_curve = MSE_curve

        self.AIF = torch.from_numpy(np.load("data/AIF.npy"))
        self.time = np.arange(0,366,2.45)
        self.pk_weight = pk_weight
        self.curve_weight = curve_weight
        
   
        
    def run_train_iter(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        output = self.model.forward(x)
        loss = self.criterion(output, y, 
                              pk_weight = self.pk_weight, curve_weight = self.curve_weight)
        
        loss_pk = self.criterion_pk(output, y, pk_weight = self.pk_weight)
        loss_curve = self.criterion_curve(output, y, curve_weight = self.curve_weight)
        

        
        loss.backward()
        self.optimizer.step()
           
        return loss.item(), loss_pk.item(), loss_curve.item()
    
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
                loss = self.criterion(output, y, 
                                      pk_weight = self.pk_weight, curve_weight = self.curve_weight)
                
                loss_pk = self.criterion_pk(output, y, pk_weight = self.pk_weight)
                loss_curve = self.criterion_curve(output, y, curve_weight = self.curve_weight)
        
        return loss.item(), loss_pk.item(), loss_curve.item()
    
    def save_model(self, model_save_dir, model_save_name, model_idx, best_validation_model_idx,
                   best_validation_model_loss):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.
        """
        if self.save:
            
            self.state['network'] = self.state_dict()  # save network parameter and other variables.
            self.state['best_val_model_idx'] = best_validation_model_idx  # save current best val idx
            self.state['best_val_model_loss'] = best_validation_model_loss  # save current best val acc
            torch.save(self.state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
                model_idx))))  # save state at prespecified filepath
        
    
    
    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network']) #loads model using pytorch function
        return state, state['best_val_model_idx'], state['best_val_model_loss']
    

    def save_statistics(self, experiment_log_dir, filename, stats_dict, current_epoch, continue_from_mode=False):
        """
        Saves the statistics in stats dict into a csv file. Using the keys as the header entries and the values as the
        columns of a particular header entry
        :param experiment_log_dir: the log folder dir filepath
        :param filename: the name of the csv file
        :param stats_dict: the stats dict containing the data to be saved
        :param current_epoch: the number of epochs since commencement of the current training session (i.e. if the experiment continued from 100 and this is epoch 105, then pass relative distance of 5.)
        :param save_full_dict: whether to save the full dict as is overriding any previous entries (might be useful if we want to overwrite a file)
        :return: The filepath to the summary file
        """
        summary_filename = os.path.join(experiment_log_dir, filename)
        mode = 'a' if continue_from_mode else 'w' #append unless its the header
        with open(summary_filename, mode) as f:
            writer = csv.writer(f)
            
            #writes the header
            if not continue_from_mode:
                writer.writerow(list(stats_dict.keys()))
            
            #writes the main part
            else:
                row_to_add = [value[current_epoch] for value in list(stats_dict.values())]
                writer.writerow(row_to_add)
    
        return summary_filename

    def load_statistics(self, experiment_log_dir, filename):
        """
        Loads a statistics csv file into a dictionary
        :param experiment_log_dir: the log folder dir filepath
        :param filename: the name of the csv file to load
        :return: A dictionary containing the stats in the csv file. Header entries are converted into keys and columns of a
         particular header are converted into values of a key in a list format.
        """
        summary_filename = os.path.join(experiment_log_dir, filename)

        with open(summary_filename, 'r+') as f:
            lines = f.readlines()

        keys = lines[0].split(",")
        keys = [key.strip('\n') for key in keys]
        stats = {key: [] for key in keys}
        for line in lines[1:]:
            values = line.split(",")
            for idx, value in enumerate(values):
                value = value.strip('\n')
                stats[keys[idx]].append(float(value))

        return stats
    
    def loss_plot(self):
        
        stats = self.load_statistics(self.experiment_logs, 'summary.csv')
        
        #normal loss plot
        for key, value in list(stats.items())[:2]:
            plt.plot(value, label = key)
            
        file_name = os.path.join(self.experiment_logs, 'loss')
            
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(file_name)
        plt.clf()
        
        #pk loss plot
        for key, value in list(stats.items())[2:4]:
            plt.plot(value, label = key)
            
        file_name = os.path.join(self.experiment_logs, 'pk_loss')
            
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("PK Loss")
        plt.savefig(file_name)
        plt.clf()
        
        #curve loss plot
        for key, value in list(stats.items())[4:6]:
            plt.plot(value, label = key)
            
        file_name = os.path.join(self.experiment_logs, 'curve_loss')
            
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Curve Loss")
        plt.savefig(file_name)
        plt.clf()
        
    
    def pk_dist(self, choice):
        self.model.eval()
        
        #so that i can check to see how it does on different datasets
        #mostly for the synthetic dataset because im leaving out the test set
        if choice == 'val':
            data = self.val_data
        elif choice == 'test':
            data =self.test_data
        else:
            print('wrong input use val or test. Defaulting to val')
            data = self.val_data
            choice = 'val' #for saving filename

        for i, (val_x, val_y) in enumerate(data):
            val_x = val_x[:,:,None].to(self.device)
            prediction_val = self.model.forward(val_x)
            prediction_val = prediction_val.cpu().detach().numpy()
            val_y = val_y.detach().numpy()
            
            combined_values = np.concatenate((prediction_val, val_y), axis = 1)
            
            #loop through and combine all the batches into one whole
            if i == 0:
                full_val = combined_values
            else:
                full_val = np.concatenate((full_val, combined_values))
                

        fig, ax = plt.subplots()
        hist_0 = ax.hist(full_val[:,0], label = 'Predicted', alpha = 0.5, bins = 100, range = [0,1])
        hist_3 = ax.hist(full_val[:,3], label = 'True', alpha = 0.5, bins = 100, range = [0,1])
        fig.legend()
        file_name = os.path.join(self.experiment_logs, 'E_hist_'+choice+'.png')
        if self.save:
            plt.savefig(file_name)
        plt.clf()
        
        hist_0 = np.array(hist_0[0])
        hist_3 = np.array(hist_3[0])
        res = np.abs(hist_0 - hist_3)
        res_E = res.sum()
        
        
        fig, ax = plt.subplots()
        hist_1 = ax.hist(full_val[:,1], label = 'Predicted', alpha = 0.5, range = [0,0.02], bins = 100)
        hist_4 = ax.hist(full_val[:,4], label = 'True', alpha = 0.5, range = [0,0.02], bins = 100)
        fig.legend()
        file_name = os.path.join(self.experiment_logs, 'Fp_hist_'+choice+'.png')
        if self.save:
            plt.savefig(file_name)
        plt.clf()
        
        hist_1 = np.array(hist_0[0])
        hist_4 = np.array(hist_3[0])
        res = np.abs(hist_1 - hist_4)
        res_Fp = res.sum()
        
        fig, ax = plt.subplots()
        hist_2 = ax.hist(full_val[:,2], label = 'Predicted', alpha = 0.5, bins = 100, range = [0,1])
        hist_5 = ax.hist(full_val[:,5], label = 'True', alpha = 0.5, bins = 100, range = [0,1])
        fig.legend()
        file_name = os.path.join(self.experiment_logs, 'vp_hist_'+choice+'.png')
        if self.save:
            plt.savefig(file_name)
        plt.clf()
        
        hist_2 = np.array(hist_2[0])
        hist_5 = np.array(hist_5[0])
        res = np.abs(hist_2 - hist_5)
        res_vp = res.sum()
        


        #divide by 100 cause the values are so large
        return (res_E, res_Fp, res_vp)
        
        
    def example_fit(self, y, pred_y, x):
        '''
        Input: x with shape (150), y with shape (3), normalised x with shape (150)
        Normalised data is needed to show the original scatter data
        '''
        self.model.eval()
        
        
        print(pred_y, y)
        target_curve = TwoCUM(y, self.time, self.AIF, 0)
        predicted_curve = TwoCUM(pred_y, self.time, self.AIF, 0)
        
        textstr = '     (True, Pred):\n E=({:.3f},{:.3f}) \n Fp=({:.3f},{:.3f})\n vp=({:.3f},{:.3f}) \n'.format(y[0], pred_y[0], y[1], pred_y[1], y[2], pred_y[2])
        plt.plot(self.time, target_curve, label = 'NLS Prediction', color = 'red')
        plt.plot(self.time, predicted_curve, label = 'Model Prediction', color = 'orange')
        plt.scatter(self.time, x, label= 'Real data')
        plt.legend()
        plt.text(0.02, 0.5, textstr, fontsize=11, transform=plt.gcf().transFigure)
        plt.subplots_adjust(left=0.30)
        plt.show()
        plt.clf()        


    def run_experiment(self):
        
        #this is for the mean values
        total_losses = {"train_loss": [], "val_loss": [], 
                        "pk_train_loss": [], "pk_val_loss": [], 
                        "curve_train_loss": [], "curve_val_loss": []}  # initialize a dict to keep the per-epoch metrics
        
        for epoch_idx in range(self.num_epochs):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_loss": [], "val_loss": [], 
                        "pk_train_loss": [], "pk_val_loss": [], 
                        "curve_train_loss": [], "curve_val_loss": []} #mini batch train and val loss
            
            self.current_epoch = epoch_idx
            
            #training run through
            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for idx, (x, y) in enumerate(self.train_data):  # get data batches
                    #(batchsize, seqlen) to (batchsize, seqlen, num_features)
                    
                    x = x[:,:,None].to(self.device)
                    y = y.to(self.device)
                    loss, loss_pk, loss_curve = self.run_train_iter(x=x, y=y)  # take a training iter step
                    current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                    current_epoch_losses["pk_train_loss"].append(loss_pk)
                    current_epoch_losses["curve_train_loss"].append(loss_curve)
                    #descriptive stuff to make it pretty
                    pbar_train.update(1)
                    pbar_train.set_description("batch loss: {:.4f}".format(loss))
            
            #validation run through
            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
                for x, y in self.val_data:  # get data batches
                    x = x[:,:,None].to(self.device)
                    y = y.to(self.device)
                    loss, loss_pk, loss_curve = self.run_evaluation_iter(x=x, y=y)  # run a validation iter
                    current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                    current_epoch_losses["pk_val_loss"].append(loss_pk)
                    current_epoch_losses["curve_val_loss"].append(loss_curve)
                                        
                    #descriptive stuff to make it pretty
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("batch loss: {:.4f}".format(loss))
                    
            val_mean_loss = np.mean(current_epoch_losses['val_loss'])
            val_mean_loss_curve = np.mean(current_epoch_losses['curve_val_loss'])
            val_model_loss_pk = np.mean(current_epoch_losses['pk_val_loss'])

            #update the learning rate
            self.lr_scheduler(val_mean_loss)
            
                
            #if it does better on the val data then save it
            if val_mean_loss < self.best_val_model_loss:  # save the best val loss
                self.best_val_model_loss = val_mean_loss  # set the best val model loss to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx
                self.best_val_model_loss_curve = val_mean_loss_curve
                self.best_val_model_loss_pk = val_model_loss_pk
                
                if self.save:
                    self.save_model(model_save_dir=self.experiment_saved_models,
                # save model and best val idx and best val acc, using the model dir, model name and model idx
                        model_save_name="train_model", model_idx=epoch_idx,
                        best_validation_model_idx=self.best_val_model_idx,
                        best_validation_model_loss=self.best_val_model_loss)
                
            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.
            
            if self.save:
                self.save_statistics(experiment_log_dir=self.experiment_logs, 
                                     filename='summary.csv',
                                     stats_dict=total_losses, 
                                     current_epoch=epoch_idx,
                                     continue_from_mode=True if (epoch_idx > 0) else False)  # save statistics to stats file.

                    
            out_string = "_".join(["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("\nEpoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            
            #break if the condition for earlystopping returns True
            self.early_stopping(val_mean_loss)
            if self.early_stopping.early_stop:
                break
        
        
        return self.best_val_model_loss, self.best_val_model_idx, self.best_val_model_loss_curve, self.best_val_model_loss_pk
    
    def testing(self):
        
        #choose the best val model to load
        #self.load_model(self.experiment_saved_models, 'train_model', self.best_val_model_idx)
        
        #train data example
        for i, (test_x, test_y) in enumerate(self.test_data):
            prediction_test = self.model.forward(test_x)
            
            prediction_test = prediction_test.detach().numpy()
            test_y = test_y.detach().numpy()
            test_x = test_x.detach().numpy()
            
            combined_values = np.concatenate((prediction_test, test_y), axis = 1)
            
            #loop through and combine all the batches into one whole
            if i == 0:
                full_test_y = combined_values
                full_test_x = test_x
            else:
                full_test_y = np.concatenate((full_test_y, combined_values))
                full_test_x = np.concatenate((full_test_x, test_x))
        
        #inverse transformation on the data
        full_test_x = self.scaler.inverse_transform(full_test_x)
        
        
        print('\n combined y array shape: ', full_test_y.shape, '\n x array shape: ', full_test_x.shape)
        while True:
            cont = input('write break to stop the loop: ')
            if cont == 'break':
                break
            
            j = int(input('input index int: '))
            self.example_fit(y = full_test_y[j,3:], pred_y = full_test_y[j,:3], x = full_test_x[j,:])
            
    def testing1(self):
        
        #choose the best val model to load
        self.load_model(self.experiment_saved_models, 'train_model', self.best_val_model_idx)
        
        self.model.eval()
        #train data example
        while True:
            breaker = input('write break to stop the loop: ')
            if breaker == 'break':
                break
                
            self.model.train()
                
            for i, (test_x, test_y) in enumerate(self.train_data):
                j = np.random.randint(low = 0, high = test_x.shape[0])
                test_x = test_x[:,:,None].to(self.device)
                prediction_test = self.model.forward(test_x)
                
                prediction_test = prediction_test.cpu().detach().numpy()
                test_y = test_y.detach().numpy()
                test_x = test_x.cpu().detach().numpy()
                test_x = np.squeeze(test_x)
                test_x = self.scaler.inverse_transform(test_x)
                
                self.example_fit(y = test_y[j,:], pred_y = prediction_test[j,:], x = test_x[j,:])
                break
            
            self.model.eval()
            
            for i, (test_x, test_y) in enumerate(self.val_data):
                j = np.random.randint(low = 0, high = test_x.shape[0])
                test_x = test_x[:,:,None].to(self.device)

                prediction_test = self.model.forward(test_x)
                
                prediction_test = prediction_test.cpu().detach().numpy()
                test_y = test_y.detach().numpy()
                test_x = test_x.cpu().detach().numpy()
                test_x = np.squeeze(test_x)
                test_x = self.scaler.inverse_transform(test_x)
                
                self.example_fit(y = test_y[j,:], pred_y = prediction_test[j,:], x = test_x[j,:])
                break
                
            for i, (test_x, test_y) in enumerate(self.test_data):
                j = np.random.randint(low = 0, high = test_x.shape[0])
                test_x = test_x[:,:,None].to(self.device)

                prediction_test = self.model.forward(test_x)
                
                prediction_test = prediction_test.cpu().detach().numpy()
                test_y = test_y.detach().numpy()
                test_x = test_x.cpu().detach().numpy()
                test_x = np.squeeze(test_x)
                test_x = self.scaler.inverse_transform(test_x)
                
                self.example_fit(y = test_y[j,:], pred_y = prediction_test[j,:], x = test_x[j,:])
                break

            
            
            
            
            
            
            