#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 23:39:55 2019

@author: dileepn

Quadratic equation solver
"""
# Math module
import math

# Complex math module for complex roots
import cmath

# Print terminology for input
print("This program solves quadratic equations of the form Ax^2 + Bx + C = 0")

# Get coefficients
try:
    a = float(input("Please enter the value for A: "))
    b = float(input("Please enter the value for B: "))
    c = float(input("Please enter the value for C: "))
except:
    print("Invalid input! Try again.")

# Calculate the roots
if type(a) == float and type(b) == float and type(c) == float:
    if (b**2 - 4*a*c) < 0:
        root1 = (-b + cmath.sqrt(b**2 - 4*a*c))/(2*a)
        root2 = (-b - cmath.sqrt(b**2 - 4*a*c))/(2*a)
    else:
        root1 = (-b + math.sqrt(b**2 - 4*a*c))/(2*a)
        root2 = (-b - math.sqrt(b**2 - 4*a*c))/(2*a)
    # Print the solution
    print("The roots of this quadratic equation are: {:.2f} and {:.2f}".format(root1, root2))