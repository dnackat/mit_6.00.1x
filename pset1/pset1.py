#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 19:15:23 2018

@author: dileepn

MIT 6.00x codes
"""
#%% PSET1
# Vowels
vowels = ['a','e','i','o','u']

s = 'azcbobobegghakl'

num_vowels = 0
for c in s:
    if c in vowels:
        num_vowels += 1

print("Number of vowels:", num_vowels)

# Bob
num_bob = 0
for i in range(0,len(s)-2):
    if s[i:i+3] == 'bob':
        num_bob += 1
        
print("Number of times bob occurs is:", num_bob)

# Alphabetical 
long = s[0]
max_long = s[0]
cur_len = 1
max_len = 1
for i in range(1,len(s)):
    if long[-1] <= s[i]:
        long += s[i]
        cur_len = len(long)
        if cur_len > max_len:
            max_len = cur_len
            max_long = long
    else:
        long = s[i]
        
print("Longest substring in alphabetical order is:", max_long)