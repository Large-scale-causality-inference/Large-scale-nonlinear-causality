#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 00:51:59 2021

@author: welcome
"""
import numpy as np
from sklearn import metrics


def recovery_performance(Adjs,label):
    Adjs=Adjs.copy()
    label=label.copy()
#    data=np.zeros((len(Adjs),len(Adjs[0].flatten())))
#    for i in range(len(Adjs)):
#        data[i,:]=Adjs[i].flatten()       
#    # Data Separation
    N=len(Adjs)
    auc_all = np.zeros((N), dtype=np.float32)
    
    for i in range(N):
        auc_all[i] = metrics.roc_auc_score(label[i].flatten(), Adjs[i].flatten())
    return auc_all

