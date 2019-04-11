#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:36:09 2019

@author: dileepn
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
x = np.linspace(1,1.2,100)
a = 1
n = 100
y = n*np.exp(-n*(x-a))

plt.plot(x,y,'r-')

print("max =", max(y))
print("min =", min(y))

#%%
H = np.array([[-1,-1],[-1,-5]])

print(np.linalg.eigvals(H))

#%%
# p-value calculation
n = 100
mubar = 3.28
sigma2bar = 15.95

Tn = np.sqrt(2*n/(3*sigma2bar))*(np.sqrt(sigma2bar)-mubar)

#%%
# Welsh-Satterthwaite formula calculation
sigma1sqhat = 0.1
sigma2sqhat = 0.2
barXn = 6.2
barYm = 6
n = 50
m = 50

N = (sigma1sqhat/n + sigma2sqhat/m)**2/((sigma1sqhat**2)/((n**2)*(n-1)) \
     + (sigma2sqhat**2)/((m**2)*(m-1)))

# Test statistic
T = (barXn - barYm)/np.sqrt(sigma1sqhat/n + sigma2sqhat/m)