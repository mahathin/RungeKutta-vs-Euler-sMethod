# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 20:44:38 2022

@author: Mahathi
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def f(t,y):
    return 2 - 2*y - (math.e)**(-4*t)

tn = 5
t0 = 0
y0 = 1
h = 0.01
n = int((tn-t0)/h)

def euler(f, t0, y0, tn, h):
    t = np.linspace(t0,tn,n)
    x = np.zeros(n)
    x[0] = y0
    for i in range(1,n):
        x[i] = x[i-1]+ f(t[i-1], x[i-1])*h
    return t, x
    
#plotting
t, x = euler(f, t0, y0, tn, h)    
plt.plot(t[2:],x[2:])
plt.show()    

#numerical
print('At t = ', tn)
print('y = ', x[n-1])