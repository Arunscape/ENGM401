import numpy as np
import numpy_financial as npf
import sympy as sym
import pandas as pd
import matplotlib.pyplot as plt


class FormulaError(Exception):
    """Exception raised for errors in my formulae

    Attributes:
        factor_notation: str
        message -- explanation of the error
    """

    def __init__(self, factor_notation: str, message: str):
        self.factor_notation = factor_notation
        self.message = message

    def __str__(self):
        return f"{self.factor_notation} -> {self.message}"
        

def simple_interest(**kwargs):
    fn = "simple interest"
    if 'P' not in kwargs:
        raise FormulaError(fn, "P, or principle amount not provided")
    elif 'i' not in kwargs:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif 'N' not in kwargs:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    P = kwargs['P']
    i = kwargs['i']
    N = kwargs['N']
    return P * (1 + i*N)

def single_compound(**kwargs):
    fn = "single cash flow (F/P, i, N)" 
    if 'P' not in kwargs:
        raise FormulaError(fn, "P, or principle amount not provided")
    elif 'i' not in kwargs:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif 'N' not in kwargs:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    P = kwargs['P']
    i = kwargs['i']
    N = kwargs['N']
    return P * (1 + i)**N

def single_present(**kwargs):
    fn = "single cash flow (P/F, i, N)"
    if 'F' not in kwargs:
        raise FormulaError(fn, "F, or future amount not provided")
    elif 'i' not in kwargs:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif 'N' not in kwargs:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    F = kwargs['F']
    i = kwargs['i']
    N = kwargs['N']
    return F/(1 + i)**N

def eq_compound(**kwargs):
    fn = "equal payment series (F/A, i, N)"
    if 'A' not in kwargs:
        raise FormulaError(fn, "A, or future amount not provided")
    elif 'i' not in kwargs:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif 'N' not in kwargs:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    A = kwargs['A']
    i = kwargs['i']
    N = kwargs['N']
    return A * ((1+i)**N -1) / i 
        
def eq_sinking(**kwargs):
    fn = "equal payment series (A/F, i N)"
    if 'F' not in kwargs:
        raise FormulaError(fn, "F, or future amount not provided")
    elif 'i' not in kwargs:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif 'N' not in kwargs:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    F = kwargs['F']
    i = kwargs['i']
    N = kwargs['N']
    return F * i/ ((1+i)**N - 1)

def eq_present(**kwargs):
    fn = "equal payment (P/A, i, N)"
    if 'A' not in kwargs:
        raise FormulaError(fn, "A, or future amount not provided")
    elif 'i' not in kwargs:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif 'N' not in kwargs:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    A = kwargs['A']
    i = kwargs['i']
    N = kwargs['N']

    return A * ((1+i)**N - 1) / (i * (1+i)**N)

def eq_captial_recovery(**kwargs):
    fn = "equal payment (A/P, i, N)"
    if 'P' not in kwargs:
        raise FormulaError(fn, "P, or principle amount not provided")
    elif 'i' not in kwargs:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif 'N' not in kwargs:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    P = kwargs['P']
    i = kwargs['i']
    N = kwargs['N']

    return P * (i * (1 + i)**N) / ((1 + i)**N - 1)
    

def linear_present(**kwargs):
   fn = "gradient series (P/G, i, N)"
   if 'G' not in kwargs:
       raise FormulaError(fn, "G, or the first payment at period 2 is missing")
   elif 'i' not in kwargs:
       raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
   elif 'N' not in kwargs:
       raise FormulaError(fn, "N, or number of number of interest periods is missing")

   G = kwargs['G']
   i = kwargs['i']
   N = kwargs['N']

   return G * ( (1+i)**N - i*N - 1 ) / (i**2 * (1 + i)**N)


def linear_conversion(**kwargs):
   fn = "gradient series (A/G, i, N)"
   if 'G' not in kwargs:
       raise FormulaError(fn, "G, or the first payment at period 2 is missing")
   elif 'i' not in kwargs:
       raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
   elif 'N' not in kwargs:
       raise FormulaError(fn, "N, or number of number of interest periods is missing")

   G = kwargs['G']
   i = kwargs['i']
   N = kwargs['N']

   return G * ( (1+i)**N - i*N - 1) / (i * ((1+i)**N - 1))
    
    
def geo_present(**kwargs):
    fn = "geometric gradient series (P/A_1, g, i, N)"
    if 'A1' not in kwargs:
        raise FormulaError(fn, "A1, or the first payment at period 1 is missing")
    elif 'g' not in kwargs:
        raise FormulaError(fn, "g, or the growth rate is missing")
    elif 'i' not in kwargs:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif 'N' not in kwargs:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    A1 = kwargs['A1']
    g = kwargs['g']
    i = kwargs['i']
    N = kwargs['N']

    if i == g:
        return A1 * N / (1+i)

    return A1 * (1 - (1+g)**N * (1+i)**(-N)) / (i-g)

        
def effective_rate_per_interest_period(r, M, K):
    return (1+ r/M)**(M/K) -1


print(eq_sinking(F=20000, i=0.09, N=5))
