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
        self.hidden = torch.nn.Linear(150, 1000) 
        self.bn1 = nn.BatchNorm1d(1000)
        self.hidden2 = torch.nn.Linear(1000, 200) 
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