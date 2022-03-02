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
    
def layer_dropout(in_f, out_f, p = 0.5):
        return nn.Sequential(nn.Linear(in_f, out_f),
                             nn.BatchNorm1d(out_f),
                             nn.ReLU(),
                             nn.Dropout(p = p))
        
def layer(in_f, out_f):
        return nn.Sequential(nn.Linear(in_f, out_f),
                             nn.BatchNorm1d(out_f),
                             nn.ReLU())

class NeuralNet(nn.Module):
    def __init__(self, enc_sizes):
        super(NeuralNet, self).__init__()
        self.enc_sizes = enc_sizes #list, for inputs and outputs size
        
        #zip iterates to smallest list size
        layer_list = [layer(in_f, out_f) for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        
        self.model = nn.Sequential(*layer_list)
        
        #use the last width in the last
        self.predict = torch.nn.Linear(self.enc_sizes[-1], 3)
        self.bound = nn.Sigmoid()

        
    def forward(self, x):

        x = self.model(x) 
        x = self.predict(x)
        x = self.bound(x)
        
        return x
    
class NeuralNet_Dropout(nn.Module):
    def __init__(self, enc_sizes, p = 0.5):
        super(NeuralNet_Dropout, self).__init__()
        self.enc_sizes = enc_sizes #list, for inputs and outputs size
        
        #zip iterates to smallest list size
        layer_list = [layer_dropout(in_f, out_f, p) for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        
        self.model = nn.Sequential(*layer_list)
        self.predict = torch.nn.Linear(self.enc_sizes[-1], 3)
        self.bound = nn.Sigmoid()

        
    def forward(self, x):

        x = self.model(x) 
        x = self.predict(x)
        x = self.bound(x)
        
        return x
    

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        #input shape is (batch size, seq_len, num_features)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        
        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out