import sympy as sym
from util import *

# semianually paid
initial_purchase = 996.25
coupon = 9.625/100
maturity= 10
par = 1000
M=2
N = maturity * M # because 15 to mature and semi annual
A = coupon/2 * par


i = sym.Symbol('i')
exp = eq_present(i=i, A=A, N=N) + single_present(i=i, N=N, F=par) - initial_purchase
print(exp)
print(sym.solveset(exp, i, domain=sym.S.Reals))
