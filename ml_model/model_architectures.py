#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:56:39 2022

@author: s1740929
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.hidden = torch.nn.Linear(150, 200) 
        self.bn1 = nn.BatchNorm1d(200)
        self.hidden2 = torch.nn.Linear(200, 200) 
        self.bn2 = nn.BatchNorm1d(200)
        self.hidden3 = torch.nn.Linear(200, 200)
        self.bn3 = nn.BatchNorm1d(200)
        self.hidden4 = torch.nn.Linear(200, 200)
        self.bn4 = nn.BatchNorm1d(200)
        self.hidden5 = torch.nn.Linear(200, 200)
        self.bn5 = nn.BatchNorm1d(200)
        self.predict = torch.nn.Linear(200, 3)
        self.bound = torch.nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.hidden(x)) )     
        x = F.relu(self.bn2(self.hidden2(x)))
        x = F.relu(self.bn3(self.hidden3(x)))
        x = F.relu(self.bn4(self.hidden4(x)))
        x = F.relu(self.bn5(self.hidden5(x)))
        x = self.predict(x) 
        x = self.bound(x)
        return x
    
def layer(in_f, out_f, p = 0.5):
        return nn.Sequential(nn.Linear(in_f, out_f),
                             nn.BatchNorm1d(out_f),
                             nn.ReLU(),
                             nn.Dropout(p = p))

class NeuralNet(nn.Module):
    def __init__(self, enc_sizes):
        super(NeuralNet, self).__init__()
        self.enc_sizes = enc_sizes #list, for inputs and outputs size
        
        #zip iterates to smallest list size
        layer_list = [layer(in_f, out_f) for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        
        self.model = nn.Sequential(*layer_list)
        self.bound = nn.Sigmoid()

        
    def forward(self, x):

        x = self.model(x) 
        
        x = self.bound(x)
        
        return x