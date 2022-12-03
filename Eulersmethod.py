# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 19:29:07 2022

@author: Mahathi
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def f(t,y):
    return 2 - 2*y - (math.e)**(-4*t)

tn = 100
t0 = 0
y0 = 1
h = 0.01
n = int((tn-t0)/h)

def rk4(t0,y0,tn,h):
    t = np.linspace(t0,tn,n)
    x = np.zeros(n)
    x[0] = y0
    for i in range(1,n):
        k1 = h*f(t[i-1], x[i-1])
        k2 = h*f(t[i-1] + h/2, x[i-1] + (k1*h/2))
        k3 = h*f(t[i-1] + h/2, x[i-1] + (k2*h/2))
        k4 = h*f(t[i-1] + h, x[i-1] + k3*h)
	
        k = (k1 + 2*k2 + 2*k3 +k4)/6
        
        x[i] = x[i-1] + k
    return t, x

#plotting
t, x = rk4(t0, y0, tn, h)  
plt.plot(t,x)
plt.show()    

#numerical
print('At t = ', tn)
print('y = ', x[n-1])


ax1.plot(x, np.absolute(V_analytic - V[int(t/dt),:])*100)
fig1 = plt.figure(figsize=(10,8))
ax1 = fig1.add_subplot(1,1,1)
ax1.set_title('Error between numerical and analytical solutions')
ax1.xlabel('Distance(m)')
ax1.ylabel('Error in Velocity of particles(m/s) between numerical and analytical solutions')