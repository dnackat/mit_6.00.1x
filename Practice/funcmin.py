#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:36:09 2019

@author: dileepn
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Plot of func
lam = np.sqrt(2)
x = np.linspace(-5,5,100)
y = (lam/2)*np.exp(-lam*np.abs(x))
y1 = (1/np.sqrt(2*np.pi))*np.exp(-(1/2)*x**2)
y2 = (1/np.pi)*(1/(1+x**2))

plt.figure()
plt.plot(x,y,'r-',label='Laplace')
plt.plot(x,y1,'b-', label='Gaussian')
plt.plot(x,y2,'k-', label='Cauchy')
plt.legend(loc='best')

#%% Eigenvalues
H = np.array([[-1,-1],[-1,-5]])

print(np.linalg.eigvals(H))

#%% p-value calculation
n = 100
mubar = 3.28
sigma2bar = 15.95

Tn = np.sqrt(2*n/(3*sigma2bar))*(np.sqrt(sigma2bar)-mubar)

#%% Welsh-Satterthwaite formula calculation
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

x_n = [0.28, 0.2, 0.01, 0.8, 0.1]

n = len(x_n)

x_n.sort()

T_n_ks = max([max(abs(i/n - x_n[i]), abs((i+1)/n - x_n[i])) for i in range(n)])

#%% Kolmogorov-Lilliefors test statistics calculation

import scipy.stats as st

x_n = [0.28, 0.2, 0.01, 0.8, 0.1]

n = len(x_n)

x_n.sort()

mu_bar = sum(x_n)/n

s_bar = sum([(x_n[i]-mu_bar)**2 for i in range(len(x_n))])/n

y_n = [st.norm(mu_bar,np.sqrt(s_bar)).cdf(x_n[i]) for i in range(len(x_n))]

T_n_kl = max([max(abs(i/n - y_n[i]), abs((i+1)/n - y_n[i])) for i in range(n)])

#%% K-S test quantile using simulation
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

#%% QQ-plot
x = [1/5,2/5,3/5,4/5,1.0]
y = [1/5,2/5,3/5,4/5,1.0]

slope, intercept = np.polyfit(x, y, 1)

abline_values = [slope * i + intercept for i in x]

x1 = [0.01,0.1,0.2,0.28,0.8]

plt.figure()
plt.plot(np.transpose(x),abline_values,'r-')
plt.plot(x,x1,'bo')

#%% Plot func
theta = 1/np.sqrt(2)

x1 = np.linspace(-1,0,50)
x2 = np.linspace(0,1,50)

y1 = [theta**2 for i in range(len(x1))]
y2 = [1-theta**2 for i in range(len(x2))]

plt.figure()
plt.plot(x1,y1,'r-', linewidth=2)
plt.plot(x2,y2,'b-.',linewidth=2)


#%% Bayes' calc mid-term
lambda_list = [0.4, 0.7]

prob = [1/5, 4/5]

post_prob_unnorm = [0, 0]

post_prob = [0, 0]

post_prob_unnorm = [prob[i]*((lambda_list[i])**3*(1-lambda_list[i])**3) \
                    for i in range(len(lambda_list))]
    
post_prob = [post_prob_unnorm[i]/sum(post_prob_unnorm) for i in \
             range(len(lambda_list))]

lambda_Bayes = sum([lambda_list[i]*post_prob[i] for i in range(len(lambda_list))]) 
    
#%% Monte-Carlo estimate of pi
import scipy.stats as st

nsim = int(1e6)

x = np.random.uniform(-1,1,nsim)
y = np.random.uniform(-1,1,nsim)

PI = 4*sum(x**2 + y**2 < 1)/nsim

print("Simulated value of pi is:", PI)

#%% Factorial and combinations
def fact(x):
    if x == 0 or x == 1:
        return 1
    
    return x*fact(x-1)

def nCx(n, x):
    if n == 0:
        return 0 
    elif x == 0:
        return 1
    elif x == 1:
        return n
    
#%% Plot func.: conditional quantile
x = np.linspace(0,1,100)

alpha1 = 0.05
alpha2 = 0.1

y1 = -x + np.sqrt(x**2 + 2*(x+0.5)*(1-alpha1))
y2 = -x + np.sqrt(x**2 + 2*(x+0.5)*(1-alpha2))

plt.figure()
plt.plot(x, y1, 'r-', linewidth = 2, label= 'alpha = 0.05')
plt.plot(x, y2, 'b-', linewidth = 2, label = 'alpha = 0.1')
plt.legend(loc = 'best')

#%% Plot functions to compare
x1 = np.linspace(-5,5,100)
x2 = np.linspace(0.1,5,100)
x3 = np.linspace(0.1,0.99,100)

y1 = x1
y2 = -1/x2
y3 = x1**2
y4 = np.log(x3**3/(1-x3**3))
y5 = -np.log(-np.log(x2/6))

plt.figure()
plt.plot(x1,y1,'k-',label='f1',linewidth=2)
plt.plot(x2,y2,'b-',label='f2',linewidth=2)
plt.plot(x1,y3,'g-',label='f3',linewidth=2)
plt.plot(x3,y4,'y-',label='f4',linewidth=2)
plt.plot(x2,y5,'c-',label='f5',linewidth=2)
plt.legend(loc='best')
plt.show()

#%% Plot func to check concavity
beta = np.linspace(-10,10,1000)

L2 = -1 - np.exp(beta)/(1 + np.exp(beta))**2

plt.plot(beta,L2,'b-',label='f1',linewidth=2)

#%% Phase noise plots
theta = np.linspace(0, np.pi/2, 500)

# Phase shift
phi_o = 0
phi_f = np.pi/3

# Before phase shift
y1_o = np.cos(theta + phi_o)    
y2_o = np.sin(theta + phi_o)

# After 30deg phase shift
y1_f = np.cos(theta + phi_f)    
y2_f = np.sin(theta + phi_f)

# Fix a theta to plot vectors before and after phase shift
theta_v = np.pi/6
origin = [0], [0]

# Vectors before and after phase shift
x_o = np.array([[np.cos(theta_v + phi_o)],[np.sin(theta_v + phi_o)]])
x_f = np.array([[np.cos(theta_v + phi_f)],[np.sin(theta_v + phi_f)]])

plt.figure()
plt.subplot(1,2,1); plt.plot(theta,y1_o,'b-',label='cosine',linewidth=2); plt.title('Signal');\
plt.grid()
plt.subplot(1,2,1); plt.plot(theta,y2_o,'r-',label='sine',linewidth=2)
plt.subplot(1,2,1); plt.plot(theta,y1_f,'b--',label='shifted cosine',linewidth=1)
plt.subplot(1,2,1); plt.plot(theta,y2_f,'r--',label='shifted sine',linewidth=1); plt.legend(loc='best')
plt.subplot(1,2,2); plt.quiver(*origin, x_o[0,0], x_o[1,0], color=['k'], \
           label = 'before shift', units='xy' ,scale=2); plt.title('Signal vector'); \
           plt.grid(); plt.xlim(-1,1); plt.ylim(-1,1)
plt.subplot(1,2,2); plt.quiver(*origin, x_f[0,0], x_f[1,0], color=['g'], \
           label = 'after shift', units='xy' ,scale=2); plt.legend(loc='lower left')
plt.savefig('phaseNoise.pdf')

#%% Trial PCA algorithm

# Generate observations drawn from a normal and build design matrix
mean = 0    # mean of distribution
variance = 1    # variance of distribution
p = 10   # features
n = 500  # samples

X = np.zeros((n,p))     # Design matrix

for i in range(n):
    X[i,:] = np.random.normal(mean, np.sqrt(variance), p)

# Compute the empirical covariance matrix, S, of X
ones = np.ones((n,1))

H = np.identity(n) - (1/n)*ones.dot(ones.T) # Orthogonal projection matrix H 

S = (1/n)*X.T.dot(H).dot(X)

# Spectral decomposition of S to get eigenvector matrix, P
P = np.linalg.svd(S)[0]

# Choose number of principal directions to keep
k = 2

# Chop P so it has only k eigenvectors (columns)
P_k = P[:,0:k]

# Project the observations onto this principal component
Y = X.dot(P_k)

# Plot the projected point cloud
plt.figure()
if k == 1:
    plt.plot(Y[:,0], np.zeros((n,1)), color='red', marker='o', markersize=4, ls = '')
    plt.xlabel('PC1')
else:
    plt.plot(Y[:,0], Y[:,1], color='red', marker='o', markersize=6, ls = '')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
plt.title('Principal Component Analysis')