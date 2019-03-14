#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:36:09 2019

@author: dileepn
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4,4,100)
y = (1/3)*x**3 - x**2 - 3*x + 10

plt.plot(x,y,'r-')

print("max =", max(y))
print("min =", min(y))