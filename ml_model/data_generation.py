#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:41:40 2022

@author: s1740929
"""
'''This is for the uterus '''
import numpy as np
import scipy.stats as st
import torch
import torch.utils.data as Data
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from tkmodel.TwoCUM_copy import TwoCUMfittingConc
from tkmodel.TwoCUM_copy import TwoCUM

class tissue():
    '''
    Originally added because i wanted to add data generation for other tissue types
    '''
    def __init__(self):
        pass
    
    @staticmethod
    def create_dataloader(x, y, batch_size, shuffle = True):
    
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
    
        
        torch_dataset = Data.TensorDataset(x, y) 
    
        dataloader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle) 
        
        return dataloader
    
    @staticmethod
    def normalise(x):
        x_trans = ( (x - x.mean() )/ x.std())
        return x_trans


#inherits from tissue, specifically for uterus
class uterus(tissue):
    
    real_x = np.load('trained_models/data/real_x_cleaned.npy')
    real_y = np.load('trained_models/data/real_y_cleaned.npy')

    def __init__(self, num, smooth = False):
        super(uterus, self).__init__()
        
        self.smooth = smooth
        self.num = num
        self.x, self.y = self.generate_xy()
        '''
        nan_ind = np.array(np.where(np.isnan(self.x)))[0]
        nan_ind = np.unique(nan_ind)
        print(nan_ind)
        self.x = np.delete(self.x, nan_ind)
        self.y = np.delete(self.y, nan_ind)
        '''
        #removes all rows that have nan values
        self.y = self.y[~np.isnan(self.x).any(axis=1)]
        self.x = self.x[~np.isnan(self.x).any(axis=1)]
        
        nan_ind = np.array(np.where(np.isnan(self.x)))[0]
        print('nan: ', nan_ind)
        
        big_ind = np.array(np.where(np.abs(self.x) > 100))[0]
        print('>100: ', big_ind)
        
        

        
    def save_data(self, experiment_folder, name):
        '''
        Because the data is randomnly generated, sometime i need to save it and load it
        '''

        path = os.path.join(experiment_folder, 'data')
        path = os.path.join(path, name)
        
        if not os.path.exists(path):  # If experiment directory does not exist
            os.mkdir(path)  # create the experiment directory
        
        path_x = os.path.join(path, name + '_x')
        path_y = os.path.join(path, name + '_y')
        np.save(path_x, self.x)
        np.save(path_y, self.y)
        
    def load_data(self, experiment_folder, name):
        '''
        Because the data is randomnly generated, sometime i need to save it and load it
        THIS WILL OVERWRITE YOUR INITIAL CHOICE OF NUM
        '''
        path = os.path.join(experiment_folder, 'data')
        path_x = os.path.join(path, name + '_x')
        path_y = os.path.join(path, name + '_y')
        self.x = np.load(path_x + '.npy')
        self.y = np.load(path_y + '.npy')
        self.num = self.x.shape[0]

        
    def E_distribution(self):
        start_percent, end_percent = 0.2157, 0.089
        params = (1.7162775627078726, 0.6992052384902265, 7.607697024682173e-06, 0.08199239222931205)
        start_nums = int(start_percent*self.num)
        end_nums = int(end_percent*self.num)
        
        start = np.random.uniform(low = 0, high= 0.00001, size = start_nums)
        end = np.random.uniform(low = 0.99, high= 1, size = end_nums)
        
        dist_num = self.num - start_nums - end_nums
        gen_data_E = st.exponweib.rvs(params[0], params[1], params[2], params[3], size=dist_num)
        gen_data_E = np.array(gen_data_E)
        
        true_dist = np.concatenate((start, gen_data_E), axis=None)
        true_dist = np.concatenate((true_dist, end), axis=None)
        
        true_dist[true_dist> 1] = 1
        
        return true_dist
    
    def Fp_distribution(self):
        params = (1.222754777103586, 0.49935926672960007, 3.0005798742817294e-05, 0.001543309761969799)
    
        gen_Fp_list = st.exponweib.rvs(params[0], params[1], params[2], params[3], size=self.num)
        true_dist = np.array(gen_Fp_list)
        
        return true_dist
    
    def vp_distrubition(self):
        start_percent, end_percent = 0.16145, 0.0839
        params = (0.501198003887801, 1.3306579742427025, 0.010005660986937497, 0.395689314967677)
        start_nums = int(start_percent*self.num)
        end_nums = int(end_percent*self.num)
        
        start = np.random.uniform(low = 0, high= 0.01, size = start_nums)
        end = np.random.uniform(low = 0.98, high= 0.99, size = end_nums)
        
        dist_num = self.num - start_nums - end_nums
        gen_data_vp = st.exponweib.rvs(params[0], params[1], params[2], params[3], size=dist_num)
        gen_data_vp = np.array(gen_data_vp)
        
        true_dist = np.concatenate((start, gen_data_vp), axis=None)
        true_dist = np.concatenate((true_dist, end), axis=None)
        
        true_dist[true_dist> 1] = 1
        
        return true_dist
    
    def E_distribution_smooth(self):
        start_percent = 0.2157
        params = (1.7162775627078726, 0.6992052384902265, 7.607697024682173e-06, 0.08199239222931205)
        start_nums = int(start_percent*self.num)
        
        start = np.random.uniform(low = 0, high= 0.00001, size = start_nums)
        
        dist_num = self.num - start_nums
        gen_data_E = st.exponweib.rvs(params[0], params[1], params[2], params[3], size=dist_num)
        gen_data_E = np.array(gen_data_E)
        
        true_dist = np.concatenate((start, gen_data_E), axis=None)
        
        true_dist[true_dist> 1] = 1
        
        return true_dist
    
    def Fp_distribution_smooth(self):
        params = (1.222754777103586, 0.49935926672960007, 3.0005798742817294e-05, 0.001543309761969799)
    
        gen_Fp_list = st.exponweib.rvs(params[0], params[1], params[2], params[3], size=self.num)
        true_dist = np.array(gen_Fp_list)
        
        return true_dist
    
    def vp_distrubition_smooth(self):
        start_percent = 0.16145
        params = (0.501198003887801, 1.3306579742427025, 0.010005660986937497, 0.395689314967677)
        start_nums = int(start_percent*self.num)
        
        start = np.random.uniform(low = 0, high= 0.01, size = start_nums)
        
        dist_num = self.num - start_nums
        gen_data_vp = st.exponweib.rvs(params[0], params[1], params[2], params[3], size=dist_num)
        gen_data_vp = np.array(gen_data_vp)
        
        true_dist = np.concatenate((start, gen_data_vp), axis=None)
        
        
        return true_dist
    
    def generate_xy(self):
        '''
        Inputs: nothing
        Outputs: x with shape (n, 150) and y with shape (n, 3)
        '''
        AIF = np.load('data/AIF.npy')
        data_size = AIF.shape[0]
        t = np.arange(0,366,2.45)
    
        #E = np.random.rand(1,num_curves) #0 to 1 for both E and vp
        #vp = np.random.rand(1,num_curves)
        #Fp = np.random.rand(1,num_curves) #this will be multplied by 1e-5 to get the right scale at the end
        if self.smooth:
            E = self.E_distribution_smooth() 
            Fp = self.Fp_distribution_smooth()
            vp = self.vp_distrubition_smooth()
        else:
            E = self.E_distribution() 
            Fp = self.Fp_distribution()
            vp = self.vp_distrubition()
    
        E = E[None,:]
        Fp = Fp[None,:]
        vp = vp[None,:]
    
    
    
        
        E_Fp = np.concatenate((E, Fp), axis =0)
        y = np.concatenate((E_Fp, vp), axis =0)
    
    
        x = np.zeros((self.num, data_size))
        for i in range(self.num):
            x[i] = TwoCUM(y[:,i], t, AIF, 0)
            if x[i].max() > 100:
                print(i, ' : ', y[:,i])
    
        y = y.T 
        
        return x,y
    
    def add_noise(self):
        '''
        Input: x with shape (n, 150)
        Output: x_noisy with shape (n, 150)
        '''
        print('x shape: ', self.x.shape)
        print('Synth max, min, mean: ', np.max(self.x), np.min(self.x), np.mean(self.x) )
        
        std_mean = 0.015483295177018167
        noise = np.random.normal(scale = 3.7*std_mean, size =self.x.shape)
        x_noisy = np.zeros(self.x.shape)
        for i in range(self.x.shape[0]):
            curve_max = np.max(self.x[i])
            x_noisy[i] = self.x[i] + curve_max * noise[i]
        
        self.x = x_noisy
        
        print('Synth noise max, min, mean: ', np.max(self.x), np.min(self.x), np.mean(self.x) )
        
    
    def plot_data(self, j = 0):
        
        t = np.arange(0,366,2.45)
        plt.scatter(t, self.x[j], label = 'Curve ' + str(j))   
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('concentration')
        plt.show()
    
    def plot_both(self):
        '''
        plots both noisy and non-noisy versions
        '''
        print('Choose a curve between 0 and ', str(self.num))
        j = int(input(': '))
        t = np.arange(0,366,2.45)
        plt.plot(t, self.x[j], label = 'Curve ' + str(j))   
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('concentration')
        
        self.add_noise()
        plt.scatter(t, self.x[j], label = 'Noisy Curve ' + str(j))   
        plt.show()        

        
    def return_dataloader(self, batch_size, shuffle = True):
        
        norm_x = uterus.normalise(self.x)
        dataloader = uterus.create_dataloader(norm_x, self.y, batch_size, shuffle)
        
        return dataloader
    
    def create_mixed_data(self, batch_train, batch_test, shuffle):
        '''
        Needs to output 3 shuffled standardised dataloaders
        '''
        scaler = StandardScaler()
        
        real_x = uterus.real_x
        real_y = uterus.real_y
        
        #remove the test set
        X_real, X_test, y_real, y_test = train_test_split(real_x, real_y, test_size = 0.2 ,shuffle = False)
        
        self.add_noise()   
        
        #split both synth and real data
        X_train_synth, X_val_synth, y_train_synth, y_val_synth = train_test_split(self.x, self.y, test_size = 0.2 ,shuffle = False)
        X_train_real, X_val_real, y_train_real, y_val_real = train_test_split(X_real, y_real, test_size = 0.2 ,shuffle = False)
        
        #combine into train and val
        X_train  = np.concatenate((X_train_synth, X_train_real), axis = 0)
        y_train  = np.concatenate((y_train_synth, y_train_real), axis = 0)
        
        X_val  = np.concatenate((X_val_synth, X_val_real), axis = 0)
        y_val  = np.concatenate((y_val_synth, y_val_real), axis = 0)
        
        
        scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        train = uterus.create_dataloader(X_train, y_train, batch_train, shuffle)
        val = uterus.create_dataloader(X_val, y_val, batch_test, shuffle)
        test = uterus.create_dataloader(X_test, y_test, batch_test, shuffle)
        
        return train, val, test, scaler
    
    def create_synth_data(self, batch_train, batch_test, shuffle):
        '''
        Needs to output 3 shuffled standardised dataloaders
        '''
        scaler = StandardScaler()
        
        real_x = uterus.real_x
        real_y = uterus.real_y
        
        #remove the test set
        X_real, X_test, y_real, y_test = train_test_split(real_x, real_y, test_size = 0.2 ,shuffle = False)
        
        self.add_noise()   
        
        #split both synth and real data
        X_train_synth, X_val_synth, y_train_synth, y_val_synth = train_test_split(self.x, self.y, test_size = 0.2 ,shuffle = False)
        
        #combine into train and val

        
        
        scaler.fit(X_train_synth)
        
        #using real instead of the test set here, because its larger and part of the same dataset
        X_train = scaler.transform(X_train_synth)
        X_val = scaler.transform(X_val_synth)
        X_real = scaler.transform(X_real)
        
        train = uterus.create_dataloader(X_train, y_train_synth, batch_train, shuffle)
        val = uterus.create_dataloader(X_val, y_val_synth, batch_test, shuffle)
        real = uterus.create_dataloader(X_real, y_real, batch_test, shuffle)
        
        return train, val, real, scaler

        
        

        
        
        
        
        
        
        
        