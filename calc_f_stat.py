#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 00:20:06 2021

@author: welcome
"""



def calc_f_stat(RSS_R,RSS_U,n,pu,pr):
    f_GC = ((RSS_R-RSS_U)/(RSS_U))*((n-pu-1)/(pu-pr));
    return f_GC