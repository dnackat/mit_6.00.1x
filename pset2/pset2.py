#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 22:26:25 2018

@author: dileepn

MIT 6.00.1x: PSET2

"""
def creditBalance(balance, annualInterestRate, monthlyPaymentRate, t):
    """ This function computes the balance of a credit card at the end of
        't' years. 
        
        Inputs:
        t: time in years as an integer or a float
        balance: current balance of cedit card as a float
        annualInterestRate: can be an integer or a float
        monthlyPaymentRate: minimum monthly payment rate - can be an integer or a float
        
        Returns credit card balance at the end of 't' years as a float rounded
        off to 2 decimal places.
        """
    # Convert t to months
    t = int(t * 12)
    
    # Calculate monthly interest rate
    mon_int_rate = annualInterestRate / 12.0
    
    # Start a loop to calculate the final balance
    for i in range(t):
        
        # Minimum monthly payment
        min_mon_payment = monthlyPaymentRate * balance
        
        # Monthly unpaid balance
        mon_unpaid_bal = balance - min_mon_payment
        
        # Update current balance
        balance = mon_unpaid_bal + mon_int_rate * mon_unpaid_bal
        
    # Return the final balance rounded to 2 decimal places
    return round(balance, 2) 

            
def fixedMonthlyPayment(balance, annualInterestRate, t):
    """ This function computes the fixed monthly payment required to pay off a
        given balance amount under 't' years. 
        
        Inputs:
        balance: current balance of cedit card as a float
        annualInterestRate: can be an integer or a float
        t: time in years as an integer or a float

        
        Returns fixed monthly payment required to pay off the balance as an 
        integer rounded off to the nearest 10. 
        """
    # The strategy adopted is to start with $10 and see if there's outstanding
    # balance at the end of 't' years. If yes, start again with $20, and repeat.
    
    # Convert t to months
    t = int(t * 12)
    
    # Calculate monthly interest rate
    mon_int_rate = annualInterestRate / 12.0
    
    # Starting value of fixed monthly payment to try
    fixed_mon_payment = 10
    
    # We need a temporary variable to store balance for each try
    temp_bal = balance
    
    # Helper function to calculate balance at the end of 't' months
    def creditBalance(balance, mon_int_rate, fixed_mon_payment, t):
    
        # Start a loop to calculate the final balance
        for i in range(t):
            
            # Monthly unpaid balance
            mon_unpaid_bal = balance - fixed_mon_payment
            
            # Update current balance
            balance = mon_unpaid_bal + mon_int_rate * mon_unpaid_bal
            
        # Return the final balance
        return balance
    
    # Start loop
    count = 0
    while temp_bal > 0:
        count += 1
        print("Guess number:", count)
        temp_bal = creditBalance(balance, mon_int_rate, fixed_mon_payment, t)
        if temp_bal <= 0:
            return fixed_mon_payment
        else:
            fixed_mon_payment += 10


def MonPaymentFast(balance, annualInterestRate):
    """ This function uses bisetion search to calculate the fixed monthly 
        payment required to pay off a given balance amount under one year. 
    
        Inputs:
        balance: current balance of cedit card as a float
        annualInterestRate: can be an integer or a float 
    
        Returns fixed monthly payment required to pay off the balance as a float
        rounded off to the nearest cent.
    """
    
    # Helper function to calculate balance at the end of 12 months
    def creditBalance(balance, mon_int_rate, fixed_mon_payment):
    
        # Start a loop to calculate the final balance
        for i in range(12):
            
            # Monthly unpaid balance
            mon_unpaid_bal = balance - fixed_mon_payment
            
            # Update current balance
            balance = mon_unpaid_bal + mon_int_rate * mon_unpaid_bal
            
        # Return the final balance
        return balance
    
    # Calculate monthly interest rate
    mon_int_rate = annualInterestRate / 12.0
    
    # Calculate lower and upper bounds of monthly payments
    lowerBound = balance / 12.0
    upperBound = (balance * (1 + mon_int_rate)**12) / 12.0
    
    # Calculate mid
    mid = (lowerBound + upperBound) / 2.0
    
    # Start loop
    count = 0
    while True:
        count += 1
        print("Guess number:", count)
        temp_bal = creditBalance(balance, mon_int_rate, mid)
        if abs(temp_bal) <= 0.01:
            return round(mid, 2)
            break
        elif temp_bal > 0.01:
            lowerBound = mid
            mid = (lowerBound + upperBound) / 2.0
        elif temp_bal < -0.01:
            upperBound = mid
            mid = (lowerBound + upperBound) / 2.0