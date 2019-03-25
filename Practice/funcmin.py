#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:36:09 2019

@author: dileepn
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1],[2],[3]])
m = 3
y = np.sum(np.abs(x-m))

plt.plot(x,y,'r-')

print("max =", max(y))
print("min =", min(y))

H = np.array([[-1,-1],[-1,-5]])

print(np.linalg.eigvals(H))
