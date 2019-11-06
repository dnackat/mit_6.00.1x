#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:52:44 2019

@author: dileepn

Playing with the Collatz conjecture.
"""
import matplotlib.pyplot as plt

def collatz(n, tries):
    """
    This function implements the collatz conjecture. Start with a number, n.
    - If n is even, halve it
    - If n is odd, compute 3*n + 1
    Continue this in a loop until we reach one.
    Inputs: 
        n: A natural number to start with
        tries: Number of tries permitted. This is to terminate in case there is 
        an infinite loop
    
    Outputs: 
        string: 'Successful' or 'Unsuccessful'.
        plot: Plot of the progress
    """
    
    counter = 0
    prog_list = []
    
    while counter <= tries:
        
        prog_list.append(n)
        
        if (n == 1):
            print("Successful. Reached 1 in {} steps.".format(counter))
            break
    
        if (n % 2 == 0):
            n = int(n/2)
        else:
            n = 3*n + 1
        
        counter += 1
    
    if counter > tries:
        print("Unsuccessful. Could not reach 1 in {} steps.".format(tries))
        
    plt.plot(range(len(prog_list)), prog_list, 'r--')
    plt.grid(axis='both')
    plt.show()