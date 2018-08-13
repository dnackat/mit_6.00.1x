#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:51:54 2018

@author: dileepn

Python Shenanigans

"""

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
    if (type(a) != int) or (type(b) != int):
        raise Exception("Need integer inputs! Usage: f(int, int)")
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

#%% Control input
data = []

file_name = input("Provide the name of the file: ")

try:
    fh = open(file_name, 'r')
except IOError:
    print("Cannot open", file_name)
else:
    # Add file contents to data
    for new in fh:
        # Remove trailing '\n'
        if new != '\n':
            addIt = new[:-1].split(',')
            data.append(addIt)
finally:
    # Close file even if writing fails
    if fh:
        fh.close()

#%% OO code: Coordiante class
class Coordinate(object):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, other):
        diff_x_sq = (self.x - other.x)**2
        diff_y_sq = (self.y - other.y)**2
        
        return (diff_x_sq + diff_y_sq)**0.5
    
    def __str__(self):
        return "<" + str(self.x) + "," + str(self.y) +">"
    
    def __sub__(self, other):
        return Coordinate(self.x - other.x, self.y - other.y)
    
#%% OO code: Fractions
class Fraction(object):
    
    def __init__(self, numer, denom):
        self.numer = numer
        self.denom = denom
    
    # Print
    def __str__(self):
        return str(self.numer) + "/" + str(self.denom)
    
    # Getters
    def getNumer(self):
        return self.numer
    
    def getDenom(self):
        return self.denom
    
    # Methods
    def __add__(self, other):
        new_num = other.getDenom() * self.getNumer() + other.getNumer() * \
                    self.getDenom()
        new_denom = self.getDenom() * other.getDenom()
        
        return Fraction(new_num, new_denom)
    
    def __sub__(self, other):
        new_num = other.getDenom() * self.getNumer() - other.getNumer() * \
                    self.getDenom()
        new_denom = self.getDenom() * other.getDenom()
        
        return Fraction(new_num, new_denom)
    
    def convert(self):
        return self.getNumer() / self.getDenom()

#%% OO code: Set of integers
class intSet(object):
    
    def __init__(self):
        self.vals = []
        
    def insert(self, e):
        if e not in self.vals:
            self.vals.append(e)
            
    def member(self, e):
        return e in self.vals
    
    def remove(self, e):
        try:
            self.vals.remove(e)
        except:
            raise ValueError(str(e) + " not found")
    
    def __str__(self):
        self.vals.sort()
        
        result = ''
        for e in self.vals:
            result += str(e) + ', '
            
        return '{' + result[:-2] + '}' 
    
    def __len__(self):
        """Returns the size of self"""
        return len(self.vals)
    
    def intersect(self, other):
        """Returns intersections of self and other"""
        to_ret = intSet()
        int_set = []
        for el in self.vals:
            if el in other.vals:
                int_set.append(el)
                to_ret.insert(el)
                
        return to_ret

#%% OO code: Animals and inheritance
class Animal(object):
    
    def __init__(self, age):
        self.age = age
        self.name = None
     
    # Getters
    def get_age(self):
        return self.age
    
    def get_name(self):
        return self.name
    
    # Setters
    def set_age(self, newAge):
        self.age = newAge
        
    def set_name(self, newName=""):
        self.name = newName
        
    def __str__(self):
        return "animal: " + str(self.name) + ", " + str(self.age)
    
# Inheritance
class cat(Animal):
    # Inherits all props of Animals
    
    def speak(self):
        print("meow")
        
    def __str__(self):
        return "cat: " + str(self.name) + ", " + str(self.age)
    
class Rabbit(Animal):
    # Example of class keeping track of the tag
    tag = 1
    
    def __init__(self, age, parent1 = None, parent2 = None):
        Animal.__init__(self, age)
        self.parent1 = parent1
        self.parent2 = parent2
        self.rid = Rabbit.tag
        Rabbit.tag += 1
    
    def speak(self):
        print("meep")
        
    def __str__(self):
        return "rabbit:"+str(self.name)+":"+str(self.age)
    
class Person(Animal):
    def __init__(self, name, age):
        Animal.__init__(self, age)
        Animal.set_name(self, name)
        self.friends = []
        
    def get_friends(self):
        return self.friends
    
    def add_friend(self, fname):
        if fname not in self.friends:
            self.friends.append(fname)
            
    def speak(self):
        print("hello")
        
    def age_diff(self, other):
        # alternate way: diff = self.age - other.age
        diff = self.get_age() - other.get_age()
        if self.age > other.age:
            print(self.name, "is", diff, "years older than", other.name)
        else:
            print(self.name, "is", -diff, "years younger than", other.name)
            
    def __str__(self):
        return "person:"+str(self.name)+":"+str(self.age)
    
import random

class Student(Person):
    # Inherits props from the Person class, which itself inherits from Animal
    def __init__(self, name, age, major=None):
        Person.__init__(self, name, age)
        self.major = major
        
    def change_major(self, major):
       self.major = major
       
    def speak(self):
        r = random.random()
        if r < 0.25:
            print("i have homework")
        elif 0.25 <= r < 0.5:
            print("i need sleep")
        elif 0.5 <= r < 0.75:
            print("i should eat")
        else:
            print("i am watching tv")
            
    def __str__(self):
        return "student:"+str(self.name)+":"+str(self.age)+":"+str(self.major)

#%% OO code: Example of inheritance
class Spell(object):
    def __init__(self, incantation, name):
        self.name = name
        self.incantation = incantation

    def __str__(self):
        return self.name + ' ' + self.incantation + '\n' + self.getDescription()
              
    def getDescription(self):
        return 'No description'
    
    def execute(self):
        print(self.incantation)


class Accio(Spell):
    def __init__(self):
        Spell.__init__(self, 'Accio', 'Summoning Charm')

class Confundo(Spell):
    def __init__(self):
        Spell.__init__(self, 'Confundo', 'Confundus Charm')

    def getDescription(self):
        return 'Causes the victim to become confused and befuddled.'

def studySpell(spell):
    print(spell)
    
spell = Accio()
spell.execute()
studySpell(spell)
studySpell(Confundo())