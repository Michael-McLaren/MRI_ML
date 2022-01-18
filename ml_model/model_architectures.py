#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:56:39 2022

@author: s1740929
"""

class FCCNetwork(nn.Module):
    def __init__(self, input_shape, num_output_classes, num_filters, num_layers, use_bias=False):