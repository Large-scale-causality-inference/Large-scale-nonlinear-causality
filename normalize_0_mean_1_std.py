#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 00:21:07 2021

@author: welcome
"""
import numpy as np

def normalize_0_mean_1_std(inp_series):
    inp_series=inp_series.copy()
    mean_ts=np.array([inp_series.mean(axis=1)]).transpose()
    mean_ts_mtrx = mean_ts*np.ones((1,inp_series.shape[1]));
    unb_data_mtrx = inp_series - mean_ts_mtrx
    p = np.power(unb_data_mtrx,2)
    s=np.array([p.sum(axis=1)]).transpose()
    sc=np.sqrt(s/p.shape[1])
    sc2=sc*(np.ones((1,p.shape[1])))
    nrm= np.divide(unb_data_mtrx,sc2)
    return nrm