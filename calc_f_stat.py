#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input: RSS_R,RSS_U,n,pu,pr
Output: f_GC as values of f-statistic

- "RSS_R" and unrestricted "RSS_U" models.
- "pu" and "pr" are the number of parameters to be estimated for the unrestricted 
           and restricted model, respectively. 
- "n" is the number of time-delayed vectors. 

"""



def calc_f_stat(RSS_R,RSS_U,n,pu,pr):
    f_GC = ((RSS_R-RSS_U)/(RSS_U))*((n-pu-1)/(pu-pr));
    return f_GC
