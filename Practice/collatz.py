#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:52:44 2019

@author: dileepn

Playing with the Collatz conjecture.
"""
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
    
    Output: A string. 'Successful' or 'Unsuccessful'. 
    """
    
    counter = 0
    while counter <= tries:
        counter += 1
        
        if (n == 1):
            print("Successful. Reached 1 in {} tries.".format(counter))
    
        if (n % 2 == 0):
            n = int(n/2)
        else:
            n = 3*n + 1
        
