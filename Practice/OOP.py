#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:51:54 2018

@author: dileepn

Python Shenanigans

"""
#%% OO code
class Student():
    def __init__(self, name, id):
        self.name = name
        self.id = id
    
    def ChangeID(self, id):
        self.id = id
        
    def print(self):
        print("{} - {}".format(self.name, self.id))

class Car():
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        
    def ChangeYear(self, year):
        self.year = year
        
    def print(self):
        print("{} - {} - {}".format(self.make, self.model, self.year))

#%% Decorators * and ** in python       
def print_tup(*args):
    print(args)

print_tup(2, 3, 4, 'a', 7, 9)

def print_dict(**kwargs):
    print(kwargs)
    
print_dict(a=2,b=4.5,c=97.1,d=-2)

#%% map and lambda functions
items = [1,2,3,4,5,6,7,8,9]
items_sq = list(map(lambda x: x**2, items))

#%% Print with input
prob = input("How many problems do you have? ")
print("I got", prob, "problems but Python ain't one.")

#%% Similarities between strings functions
def lines(a, b):
    """Return lines in both a and b"""
    
    a = set(a.split("\n"))
    b = set(b.split("\n"))
    
    sim_lines = list(a & b)
    
    return sim_lines 

from nltk.tokenize import sent_tokenize

def sentences(a, b):
    """Return sentences in both a and b"""
    
    a = set(sent_tokenize(a, language='english'))
    b = set(sent_tokenize(b, language='english'))

    sim_sentences = list(a & b)
    
    return sim_sentences 

 
# Define a function to return substrings of length n
def substr(a, n):
    """Return substrings of length n from string a"""
    
    # List to store substrings
    subs = []
    
    # Loop through a and create substrings of length n
    for i in range(len(a) - (n - 1)):
        subs.append(a[i:i+n])
            
    return subs

def substrings(a, b, n):
    """Return substrings of length n in both a and b"""
    
    # Get substrings of length n from a and b and store them as sets
    a = set(substr(a, n))
    b = set(substr(b, n))
    
    # Create a list of substrings that appear in both a and b
    sub_strings = list(a & b)
    
    return sub_strings

#%% Assert in python
while True:
    try:
        x = float(input("Enter a positive number: "))
    except ValueError:
        print("Please provide a numeric input.")
        continue
    break

assert (x > 0), "{} is not a positive number!".format(x)
print("You gave me ", x)

#%% Dynamic programming: Edit distance between two strings
def edit_dist(a, b):
    
    # Create a matrix to store edit distances
    ed = [[0 for x in range(len(a) + 1)] for x in range(len(b) + 1)]
    
    #  Start loop to fill dp bottom-up
    for i in range(len(a) + 1):
        for j in range(len(b) + 1):
            
            # Base case 1: If len(a) = 0, you have to do j substitutions to get b
            if len(a) == 0:
                ed[i][j] = j
            
            # Base case 2: If len(b) = 0, you have to do i deletions to get a
            elif len(b) == 0:
                ed[i][j] = i
                
            # If last characters are the same, ignore last character
            elif a[i - 1] == b[i - 1]:
                ed[i][j] = ed[i - 1][j - 1]
                
            # In all other cases, recursively find edit distance
            else:
                ed[i][j] = 1 + min(ed[i][j - 1],     # Insert
                                ed[i - 1][j],    # Remove
                                ed[i - 1][j - 1])    #Replace
        
    # Return edit distance
    return ed[len(a)][len(b)]

#%% Dynamic Programming: Fibonacci sequence
def fib_dp(n, memo={}):
    """Assumes n is an int >= 0, memo used only by recursive calls. Returns
       Fibonacci of n."""
    
    # Base cases
    if n == 0 or n == 1:
        return 1
    
    # Recursive case
    try:
        return memo[n]
    except KeyError:
        result = fib_dp(n - 1, memo) + fib_dp(n - 2, memo)
        memo[n] = result
        return result

#%% Compute derivative of a given polynomial at a given point
def derivative():
    """ Computes derivative of the give polynomial (one variable) at the given point. """
    
    # Check if degree is a positive number
    while True:
        try:
            degree = int(input("Enter the degree of polynomial: "))
        except ValueError:
            print("Degree must be a positive integer.")
            continue
        if degree < 0:
            print("Degree must be a positive integer.")
            continue
        else:
            break
        
    # If degree is 0 return 0
    if degree == 0:
        return 0
    
    # Get the coefficients and store them in a list
    coeff = []
    for i in range(degree + 1):
        coeff.append(float(input("Coeff. for degree {}: ".format(degree - i))))
    
    # Get the point at which derivative is to be calculated
    while True:
        try:
            x = float(input("Enter the point at which derivative is to be calculated: "))
        except ValueError:
            print("Please provide a numeric input.")
            continue
        break
    
    # Compute derivative
    der = 0
    for i in range(degree + 1):
        power = degree - i
        
        # If power = 0 (constant term), der = 0
        if power == 0:
            der += 0
        else:
            der += coeff[i] * power * x**(power - 1)
            
    # Return derivative at x
    return der

#%% Guessing a number: Bisection search
print("Please think of a number between 0 and 100!")

guess = 50
low = 0
high = 100
while True:
    print("Is your secret number {}? ".format(guess))
    inp = input("Enter 'h' to indicate the guess is too high. Enter 'l' to indicate the guess is too low. Enter 'c' to indicate I guessed correctly. ")
    if inp == "l":
        low = guess
        guess = int((low + high)/2)
    elif inp == "h":
        high = guess
        guess = int((low + high)/2)
    elif inp == "c":
        break
    else: 
        print("Sorry, I did not understand your input.")
print("Game over. Your secret number was:", guess)

#%% Convert decimal to binary
num = int(input("Enter a decimal number: "))

if num < 0:
    isNeg = True
    num = abs(num)
else:
    isNeg = False

result = ''
if num == 0:
    result = '0'

while num > 0:
    result = str(num % 2) + result
    num //= 2
if isNeg:
    result = '-' + result

print("Binary representation:", result)

#%% Convert float to binary
x = float(input("Enter a decimal number between 0 and 1: "))

p = 0
while (2**p)*x % 1 != 0:
    print("Remainder = " + str((2**p)*x - int((2**p)*x)))
    p += 1
    
num = int((2**p)*x)

result = ''
if num == 0:
    result = '0'

while num > 0:
    result = str(num % 2) + result
    num //= 2
    
for i in range(p - len(result)):
    result = '0' + result
    
result = result[0:-p] + "." + result[-p:]
print("Binary representation of the decimal:", str(x), "=", result)

#%% Towers of Hanoi: Recursion
def printMove(fr, to):
    print("Move from " + str(fr) + " to " + str(to))
    
def Towers(n, fr, to, spare):
    if n == 1:
        printMove(fr, to)
    else:
        Towers(n-1, fr, spare, to)
        Towers(1, fr, to, spare)
        Towers(n-1, spare, to, fr)
        
#%% Bisection search using recursion
def isIn(char, aStr):
    '''
    char: a single character
    aStr: an alphabetized string
    
    returns: True if char is in aStr; False otherwise
    '''
    if len(aStr) == 0:
        return False
    elif len(aStr) == 1:
        if aStr == char:
            return True
        else:
            return False
    elif aStr[int(len(aStr)/2)] == char:
        return True
    elif aStr[int(len(aStr)/2)] < char:
        return isIn(char, aStr[int(len(aStr)/2) + 1:])
    elif aStr[int(len(aStr)/2)] > char:
        return isIn(char, aStr[0:int(len(aStr)/2)])
    else:
        return False
#%% Polysum
from math import pi, tan

def polysum(n, s):
    # Calculate area
    area = 0.25 * n * s**2 / tan(pi/n)
    # Calculate perimeter squared
    perimeter_sq = (n * s)**2
    # Return their sum
    return round(area + perimeter_sq, 2)

#%% Tuples
def get_data(aTuple):
    nums = ()
    words = ()
    
    for t in aTuple:
        nums = nums + (t[0],)
        if t[1] not in words:
            words = words + (t[1],)
    min_nums = min(nums)
    max_nums = max(nums)
    unique_words = len(words)
    
    return (min_nums, max_nums, unique_words)