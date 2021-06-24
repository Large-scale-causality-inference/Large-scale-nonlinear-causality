#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input: RSS_R,RSS_U,n,pu,pr
Output: f_GC as values of f-statistic
The code takes in restricted "RSS_R" and unrestricted "RSS_U" models, number of parameters 
in restricted and unrestricted models as "pu" and "pr", and "n" 
is the number of time-delayed vectors. 

"""



def calc_f_stat(RSS_R,RSS_U,n,pu,pr):
    f_GC = ((RSS_R-RSS_U)/(RSS_U))*((n-pu-1)/(pu-pr));
    return f_GC
