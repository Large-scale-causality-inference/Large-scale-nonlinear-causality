#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 00:16:28 2021

@author: welcome
"""
import torch
import numpy as np


def multivariate_split(X,ar_order, valid_percent=0):
    X=X.copy()
    TS=np.shape(X)[1]
    n_vars=np.shape(X)[0]
    val_num=int(valid_percent*TS)
    my_data_train=torch.zeros((TS-ar_order-val_num,ar_order,n_vars))
    my_data_y_train=torch.zeros((TS-ar_order-val_num,1,n_vars))
    my_data_val=torch.zeros((val_num,ar_order,n_vars))
    my_data_y_val=torch.zeros((val_num,1,n_vars))
    for i in range(TS-ar_order-val_num):
        my_data_train[i]=torch.from_numpy(X.transpose()[i:i+ar_order,:])
        my_data_y_train[i]=torch.from_numpy(X.transpose()[i+ar_order,:])

    for i in range(TS-ar_order-val_num, TS-ar_order,1):
        my_data_val[i-(TS-ar_order-val_num)]=torch.from_numpy(X.transpose()[i:i+ar_order,:])
        my_data_y_val[i-(TS-ar_order-val_num)]=torch.from_numpy(X.transpose()[i+ar_order,:])
    return my_data_train, my_data_y_train, my_data_val, my_data_y_val