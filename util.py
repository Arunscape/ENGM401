import numpy as np
import numpy_financial as npf
import sympy as sym
import pandas as pd
import matplotlib.pyplot as plt
import math


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


def simple_interest(P=None, i=None, N=None):
    fn = "simple interest"
    if P is None:
        raise FormulaError(fn, "P, or principle amount not provided")
    elif i is None or i <=0 or 1 <= i:
        raise FormulaError(fn, "i, or interest not provided: 0<i<1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return P * (1 + i*N)

def single_future(P=None, i=None, N=None):
    fn = "single cash flow (F/P, i, N)"
    if P is None:
        raise FormulaError(fn, "P, or principle amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return P * (1 + i)**N

def single_present(F=None, i=None, N=None):
    fn = "single cash flow (P/F, i, N)"
    if F is None:
        raise FormulaError(fn, "F, or future amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return F/(1 + i)**N

def single_rate(F=None, P=None, N=None):
    fn = "find the discounting rate that makes a present amount equivalent to a future amount"
    if F is None:
        raise FormulaError(fn, "F, or future amount not provided")
    if P is None:
        raise FormulaError(fn, "P, or present amount not provided")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return npf.rate(N, 0, -P, F)


def single_periods(F=None, P=None, i=None):
    fn = "find the number of periods makes a present amount equivalent to a future amount"
    if F is None:
        raise FormulaError(fn, "F, or future amount not provided")
    if P is None:
        raise FormulaError(fn, "P, or present amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")

    return npf.nper(i, 0, -P, F)

def eq_future(A=None, i=None, N=None):
    fn = "equal payment series (F/A, i, N)"
    if A is None:
        raise FormulaError(fn, "A, or future amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return A * ((1+i)**N -1) / i

def eq_sinking(F=None, i=None, N=None):
    fn = "equal payment series (A/F, i N): gives future value"
    if F is None:
        raise FormulaError(fn, "F, or future amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return F * i/ ((1+i)**N - 1)

def eq_present(A=None, i=None, N=None):
    fn = "equal payment (P/A, i, N)"
    if A is None:
        raise FormulaError(fn, "A, or future amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return A * ((1+i)**N - 1) / (i * (1+i)**N)

def eq_capital_recovery(P=None, i=None, N=None):
    fn = "equal payment (A/P, i, N): gives A"
    if P is None:
        raise FormulaError(fn, "P, or principle amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return P * (i * (1 + i)**N) / ((1 + i)**N - 1)


def linear_present(G=None, i=None, N=None):
   fn = "gradient series (P/G, i, N)"
   if G is None:
       raise FormulaError(fn, "G, or the first payment at period 2 is missing")
   elif i is None:
       raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
   elif N is None:
       raise FormulaError(fn, "N, or number of number of interest periods is missing")

   return G * ( (1+i)**N - i*N - 1 ) / (i**2 * (1 + i)**N)


def linear_conversion(G=None, i=None, N=None):
   fn = "gradient series (A/G, i, N): gives A"
   if G is None:
       raise FormulaError(fn, "G, or the first payment at period 2 is missing")
   elif i is None:
       raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
   elif N is None:
       raise FormulaError(fn, "N, or number of number of interest periods is missing")

   return G * ( (1+i)**N - i*N - 1) / (i * ((1+i)**N - 1))


def geo_present(A1=None, g=None, i=None, N=None):
   fn = "geometric gradient series (P/A_1, g, i, N)"
   if A1 is None:
       raise FormulaError(fn, "A1, or the first payment at period 1 is missing")
   elif g is None:
       raise FormulaError(fn, "g, or the growth rate is missing")
   elif i is None:
       raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
   elif N is None:
       raise FormulaError(fn, "N, or number of number of interest periods is missing")

   if i == g:
       return A1 * N / (1+i)

   return A1 * (1 - (1+g)**N * (1+i)**(-N)) / (i-g)


def effective_rate_per_interest_period(r=None, M=None, K=None, C=None):
    fn = """effective_rate_per_interest_period: i_a
    r = APR or nominal interest rate per year
    M = number of compounding periods per year, M=CK
    C = number of compounding periods per payment period
    K = number of payment periods per year
    """

    if r is None:
        raise FormulaError(fn, "r, or APR, or nominal interest rate per year is missing")

    if M == math.inf and K is not None:
        return math.exp(r/K) -1

    if M == math.inf and K is None:
        return math.exp(r/K) -1

    if C is not None and K is not None and M is None:
        return (1+ r/(C*K))**C - 1

    if K is None and C is None and M is not None:
        return (1 + r/M)**M - 1

    if C is not None and K is None and M is not None:
        return (1+r/M)**C - 1

    if M is not None and K is not None and C is None and M is not None:
        return (1+ r/M)**(M/K) -1

    raise FormulaError(fn, "invalid combo of inputs")

# amortized
def remaining_balance():
    pass

def interest_payment():
    pass

def principal_repayment():
    pass
