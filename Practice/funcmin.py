#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:36:09 2019

@author: dileepn
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
eps = 0
x = np.linspace(eps,1-eps,100)
y = 1/(x*(1-x))

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

#%% Bayes' calc
lambda_list = [0.2, 0.4, 0.6, 0.8]

prob = [0.2, 0.4, 0.2, 0.2]

post_prob = [0, 0, 0, 0]

Lhood = 0

for i in range(len(lambda_list)):
    Lhood += prob[i]*((lambda_list[i])**4*(1-lambda_list[i])**2)
    
for i in range(len(lambda_list)):
    post_prob[i] = prob[i]*((lambda_list[i])**4*(1-lambda_list[i])**2)/Lhood

#%% Kolmogorov-Smirnov test statistics calculation
import math

x_n = [0.28, 0.2, 0.01, 0.8, 0.1]

n = len(x)

x_n.sort()

T_n = math.sqrt(n)*max([max(abs(i/n - x_n[i]), abs((i+1)/n - x_n[i])) for i in range(n)])

#%% K-S test using simulation
import numpy as np
from scipy import stats 
import matplotlib.pyplot as plt

# Basic constants

n = 20

def tlcalculator():
    """
    Does the optimization to find the supremum over the domain 0 <= x <= 1
    """
    nrvs = stats.uniform.rvs(loc=0, scale=1, size=n)
    nrvs.sort()
    comp_1 = np.linspace(0,1,n,False)
    comp_2 = comp_1 + 1/n

    a = np.max(abs(nrvs-comp_1))
    b = np.max(abs(nrvs-comp_2))

    return max(a,b)

# Repeat M times and then plot histogram
M = 220000
m_all = np.array([tlcalculator() for i in range(M)])
m_all.sort()

plt.hist(m_all, bins=100);

print("estimated 95% quantile is: {0} ".format(m_all[int(0.95*M)]))