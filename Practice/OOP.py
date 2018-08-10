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

#%% Odd tuples
def oddTuples(aTup):
    '''
    aTup: a tuple
    
    returns: tuple, every other element of aTup. 
    '''
    ret_tup = ()
    for i in range(0,len(aTup),2):
        ret_tup += (aTup[i],)
        
    return ret_tup

#%% Map function to list
def applyToEach(L, f):
    """ Assumes L is a list, f a function mutates the list by applying to each
        element, e, of L by f(e). """
        
    for i in range(len(L)):
        L[i] = f(L[i])
        
#%% Apply list of functions to a number
def applyFuns(L, x):
    for f in L:
        print(f(x))
        
#%% Dicts: Analyze song lyrics
def lyrics_to_frequencies(lyrics):
    """ Frequency of words. 'lyrics' is a list of strings. """
    myDict = {}
    for word in lyrics:
        if word in myDict:
            myDict[word] += 1
        else:
            myDict[word] = 1
            
    return myDict

def most_common_words(freqs):
    """ Most common words. 'freqs' is a dictionary. """
    values = freqs.values()
    best = max(values)
    words = []
    
    for k in freqs:
        if freqs[k] == best:
            words.append(k)
            
    return (words, best)

def words_often(freqs, minTimes):
    """ Words in freqs appearing atleast 'minTimes' number 
        of times. 'freqs' is a dictionary."""
    result = []
    done = False
    while not done:
        temp = most_common_words(freqs)
        if temp[1] >= minTimes:
            result.append(temp)
            for w in temp[0]:
                del(freqs[w])
        else:
            done = True
            
    return result

#%% All instances in a dict
def how_many(aDict):
    '''
    aDict: A dictionary.

    returns: int, how many values are in the dictionary.
    '''
    # Your Code Here
    count = 0
    for key in aDict.keys():
        if type(aDict[key]) == list:
            count += len(aDict[key])
        elif type(aDict[key]) == dict:
            count += len(aDict[key])
        else:
            count += 1
            
    return count

def biggest(aDict):
    '''
    aDict: A dictionary.

    returns: The key with the largest number of values associated with it
    '''

    # Your Code Here
    biggest = 0
    biggestkey = ''
    for key in aDict.keys():
        if type(aDict[key]) == list:
            if len(aDict[key]) > biggest:
                biggest = len(aDict[key])
                biggestkey = key
        elif type(aDict[key]) == dict:
            if len(aDict[key]) > biggest:
                biggest = len(aDict[key])
                biggestkey = key
            
    return biggestkey

#%% Recursive 7 count
def count7(N):
    '''
    N: a non-negative integer
    '''
    if N < 7:
        return 0
    elif N == 7:
        return 1
    else:
        if N % 10 == 7:
            return 1 + count7(N//10)
        else:
            return 0 + count7(N//10) 
        
#%% Dot product of 2 vectors
def dotProduct(listA, listB):
    '''
    listA: a list of numbers
    listB: a list of numbers of the same length as listA
    '''
    dot_prod = 0
    for i in range(len(listA)):
        dot_prod += listA[i] * listB[i]
            
    return dot_prod

#%% Unique values in a dict
def uniqueValues(aDict):
    '''
    aDict: a dictionary
    '''
    
    key_list = []
    count = {}
    
    for key in aDict.keys():
        if aDict[key] in count:
            count[aDict[key]] += 1
        else:
            count[aDict[key]] = 1
       
    for key in count.keys():
        if count[key] == 1:
            for k in aDict.keys():
                if aDict[k] == key:
                    key_list.append(k)
            
    return sorted(key_list)

#%% Flatten a list: order matters --Recursion
calls = 0
def flatten(aList):
    ''' 
    aList: a list 
    Returns a copy of aList, which is a flattened version of aList 
    '''
    global calls
    calls += 1
    
    if len(aList) == 0:
        return []
    else:
        if type(aList[0]) == list:
            return flatten(aList[0]) + flatten(aList[1:])

    return aList[:1] + flatten(aList[1:])

b = [[1, 'a', ['cat'], 2], [[[3]], 'dog'], 4, 5]

b_flat = flatten(b)

print("Original list:", b)
print("Flattened list:", b_flat)
print("I made", calls, "recursive calls.")

#%% Scoring a word
import string

def f(a, b):
    return a + b

def score(word, f):
    """
       word, a string of length > 1 of alphabetical 
             characters (upper and lowercase)
       f, a function that takes in two int arguments and returns an int

       Returns the score of word as defined by the method:

    1) Score for each letter is its location in the alphabet (a=1 ... z=26) 
       times its distance from start of word.  
       Ex. the scores for the letters in 'adD' are 1*0, 4*1, and 4*2.
    2) The score for a word is the result of applying f to the
       scores of the word's two highest scoring letters. 
       The first parameter to f is the highest letter score, 
       and the second parameter is the second highest letter score.
       Ex. If f returns the sum of its arguments, then the 
           score for 'adD' is 12 
    """

    alph = string.ascii_letters
    scores = []
    
    if len(word) == 0:
        return 0
    
    for i in range(len(word)):
        score = i * ((alph.index(word[i]) % 26) + 1)
        scores.append(score)
    
    a = max(scores)
    if len(scores) > 1:
        scores.remove(max(scores))
        b = max(scores)
    else:
        b = 0
    
    return f(a, b)