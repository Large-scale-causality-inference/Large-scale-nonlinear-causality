#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The code returns performance measures (in terms of AUROC) given a list of 
"Adjs": data matrices and 
"label": label matrices. 

@Reference: 
Wismüller, A., Dsouza, A.M., Vosoughi, M.A. et al. 
Large-scale nonlinear Granger causality for inferring directed dependence from short multivariate time-series data. Sci Rep 11, 7817 (2021).

"""
import numpy as np
from sklearn import metrics


def recovery_performance(Adjs,label):
    Adjs=Adjs.copy()
    label=label.copy()
    N=len(Adjs)
    auc_all = np.zeros((N), dtype=np.float32)
    
    for i in range(N):
        auc_all[i] = metrics.roc_auc_score(label[i].flatten(), Adjs[i].flatten())
    return auc_all

