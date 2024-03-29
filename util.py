import numpy as np
import numpy_financial as npf
import sympy as sym
import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import List
from matplotlib.ticker import MaxNLocator


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
    elif i is None or i <= 0 or 1 <= i:
        raise FormulaError(fn, "i, or interest not provided: 0<i<1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return P * (1 + i * N)


def single_future(P=None, i=None, N=None):
    fn = "single cash flow (F/P, i, N)"
    if P is None:
        raise FormulaError(fn, "P, or principle amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return P * (1 + i) ** N


def single_present(F=None, i=None, N=None):
    fn = "single cash flow (P/F, i, N)"
    if F is None:
        raise FormulaError(fn, "F, or future amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return F / (1 + i) ** N


def time_shift(i: float, N: int):
    if N > 0:
        return single_future(1, i, N)
    else:
        return single_present(1, i, -N)


def single_rate(F=None, P=None, N=None):
    fn = "find the discounting rate that makes a present amount equivalent to a future amount"
    if F is None:
        raise FormulaError(fn, "F, or future amount not provided")
    if P is None:
        raise FormulaError(fn, "P, or present amount not provided")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return npf.rate(N, 0, -P, F)


def eq_rate(A=None, P=None, N=None):
    fn = "find the discounting rate that makes equal payments equivalent to a furture amount"
    if A is None:
        raise FormulaError(fn, "A, or payment not provided")
    if P is None:
        raise FormulaError(fn, "P, or present amount not provided")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return npf.rate(N, A, -P, 0)


def single_periods(F=None, P=None, i=None):
    fn = "find the number of periods makes a present amount equivalent to a future amount"
    if F is None:
        raise FormulaError(fn, "F, or future amount not provided")
    if P is None:
        raise FormulaError(fn, "P, or present amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")

    return npf.nper(i, 0, -P, F)


def eq_periods(A=None, P=None, i=None):
    fn = "find the number of periods makes an equal payment equivalent to a future amount"
    if A is None:
        raise FormulaError(fn, "A, or payment not provided")
    if P is None:
        raise FormulaError(fn, "P, or present amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")

    return npf.nper(i, A, P, 0)


def eq_future(A=None, i=None, N=None):
    fn = "equal payment series (F/A, i, N)"
    if A is None:
        raise FormulaError(fn, "A, or future amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return A * ((1 + i) ** N - 1) / i


def eq_sinking(F=None, i=None, N=None):
    fn = "equal payment series (A/F, i N): gives future value"
    if F is None:
        raise FormulaError(fn, "F, or future amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return F * i / ((1 + i) ** N - 1)


def eq_present(A=None, i=None, N=None):
    fn = "equal payment (P/A, i, N)"
    if A is None:
        raise FormulaError(fn, "A, or future amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return A * ((1 + i) ** N - 1) / (i * (1 + i) ** N)


def eq_capital_recovery(P=None, i=None, N=None):
    fn = "equal payment (A/P, i, N): gives A"
    if P is None:
        raise FormulaError(fn, "P, or principle amount not provided")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return P * (i * (1 + i) ** N) / ((1 + i) ** N - 1)


def linear_present(G=None, i=None, N=None):
    fn = "gradient series (P/G, i, N)"
    if G is None:
        raise FormulaError(fn, "G, or the first payment at period 2 is missing")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return G * ((1 + i) ** N - i * N - 1) / (i ** 2 * (1 + i) ** N)


def linear_conversion(G=None, i=None, N=None):
    fn = "gradient series (A/G, i, N): gives A"
    if G is None:
        raise FormulaError(fn, "G, or the first payment at period 2 is missing")
    elif i is None:
        raise FormulaError(fn, "i, or interest not provided: 0<=i<=1")
    elif N is None:
        raise FormulaError(fn, "N, or number of number of interest periods is missing")

    return G * ((1 + i) ** N - i * N - 1) / (i * ((1 + i) ** N - 1))


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
        return A1 * N / (1 + i)

    return A1 * (1 - (1 + g) ** N * (1 + i) ** (-N)) / (i - g)


def effective_rate_per_payment_period(r=None, M=None, K=None, C=None):
    """
    effective_rate_per_payment_period:
    r = APR or nominal interest rate per year
    M = number of compounding periods per year, M=CK
    C = number of compounding periods per payment period
    K = number of payment periods per year
    """

    fn = """effective_rate_per_payment_period: i_a
    r = APR or nominal interest rate per year
    M = number of compounding periods per year, M=CK
    C = number of compounding periods per payment period
    K = number of payment periods per year
    """

    if r is None:
        raise FormulaError(
            fn, "r, or APR, or nominal interest rate per year is missing"
        )

    if M == math.inf and K is not None:
        return math.exp(r / K) - 1

    if M == math.inf and K is None:
        return math.exp(r) - 1

    if C is not None and K is not None and M is None:
        return (1 + r / (C * K)) ** C - 1

    if K is None and C is None and M is not None:
        return (1 + r / M) ** M - 1

    if C is not None and K is None and M is not None:
        return (1 + r / M) ** C - 1

    if M is not None and K is not None and C is None:
        return (1 + r / M) ** (M / K) - 1

    raise FormulaError(fn, "invalid combo of inputs")


def effective_annual_interest_rate(r=None, M=None):
    return effective_rate_per_payment_period(r=r, M=M)


def ia_APR_per_month_compounded_monthly(r_over_m):
    return (1 + r_over_m) ** 12 - 1


def ia(r_over_m, M=2):
    return (1 + r_over_m) ** M - 1


# amortized
def remaining_balance(A=None, i=None, N=None, n=None):
    return eq_present(A, i, N=(N - n))


def interest_payment(A=None, i=None, N=None, n=None):
    # return remaining_balance(A, i, N, n=(n - 1)) * i
    return eq_present(A, i, N - n + 1) * i


def principal_payment(P=None, i=None, N=None, n=None):
    return npf.ppmt(i, n, N, -P)


def total_interest(A=None, i=None, N=None):
    sum = 0
    for n in range(1, N + 1):
        interest = interest_payment(A, i, N, n)
        # print(f"period {n}, interest: {interest}")
        sum += interest
    return sum


def add_on_interest(P, i, N):
    return P * i * N


def bond_payment(coupon=None, par=None, M=2):
    return coupon * par / M


def YTM(purchase_price=None, coupon=None, years=None, par=None, M=2):
    # M=2 means semiannual
    N = years * M  # number of payment periods

    # division, but python doesn't do floats well...
    A = sym.Rational(coupon * par, M)

    i = sym.symbols("i")

    expr = eq_present(A=A, i=i, N=N) + single_present(F=par, i=i, N=N) - purchase_price
    print("note: this is probably the semiannual rate if M=2 i.e. you now have r/m")
    print("use ia(r_over_m, M=2) to get annual interest rate")
    print("multiply it by 2 to get the nominal annual rate")
    return solve(expr, i)


def YTM_NA(purchase_price=None, A=None, coupon=None, N=None, par=None, M=2):
    # M=2 means semiannual

    i = sym.symbols("i")
    expr = eq_present(A=A, i=i, N=N) + single_present(F=par, i=i, N=N) - purchase_price
    print("expr", expr)
    print("note: this is probably the semiannual rate if M=2 i.e. you now have r/m")
    print("use ia(r_over_m, M=2) to get annual interest rate")
    print("multiply it by 2 to get the nominal annual rate")
    return solve(expr, i)


def bond_market(A=None, coupon=None, years=None, par=None, M=2):
    N = years * M
    i = coupon / M
    return eq_present(A=A, i=i, N=N) + single_present(F=par, i=i, N=N)


def bond_current_yield(A=None, market=None):
    print("this is the semiannual yield")
    print("multiply by M=2 to get the annual current yield")
    return A / market


def solve(expr, var):
    return sym.solveset(expr, var, domain=sym.S.Reals)


# def net_present_worth(cash_flows: List[float], i: float):
#     sum: float = 0
#     for n, cf in enumerate(cash_flows):
#         sum += single_present(F=cf, i=i, N=n)
#     return sum


def net_present_worth(rate, cash_flows):
    return npf.npv(rate, cash_flows)


def project_balance(i: float, project_balance: List[float]):
    An = []
    cost = []
    difference = []

    for n, pb in enumerate(project_balance):
        diff = pb
        c = 0
        an = diff
        if n > 0:
            diff -= project_balance[n - 1]
            c = i * project_balance[n - 1]
            an = diff - c
        difference.append(diff)
        cost.append(c)
        An.append(an)

    d = {
        "An": An,
        "interest": cost,
        "project balance": project_balance,
        "difference": difference,
    }
    print("net present worth", npf.npv(i, An))
    return pd.DataFrame(data=d)


def cash_flows(i: float, cash_flows: List[float]):
    project_balance = []
    cost = []
    difference = []

    if i is None:
        i = rate_of_return(cash_flows)
        print(f"rate of return is {i}")

    for n, cf in enumerate(cash_flows):
        pb = cf
        diff = cf
        c = 0
        if n > 0:
            c = i * project_balance[n - 1]
            pb = project_balance[n - 1] + cf + c
            diff = pb - project_balance[n - 1]
        difference.append(diff)
        cost.append(c)
        project_balance.append(pb)

    d = {
        "An": cash_flows,
        "interest": cost,
        "project balance": project_balance,
        "difference": difference,
    }
    print("net present worth", npf.npv(i, cash_flows))
    print("equivalent annual worth", equivalent_annual_worth(i, cash_flows))
    return pd.DataFrame(data=d)

def equivalent_annual_worth(i: float, cash_flows: List[float]):
    # start at n=0
    sum = net_present_worth(i, cash_flows)
    A = eq_capital_recovery(sum, i, len(cash_flows)-1)
    print("equivalent annual cost, capitalized equivalent worth")
    return (A, A/i)

def capitalized_equivalent(A, i):
    return A/i

def rate_of_return(cash_flows):
    return npf.irr(cash_flows)

def capital_recovery_cost(P, i, N, Salvage):
    return eq_capital_recovery(P-Salvage, i, N) + i*Salvage


def declining_balance(P, d, n):
    dep = d * P * (1-d) ** (n-1)
    book = P * (1 - d) ** n 
    print("(depreciation, book value)")
    return (dep, book)

def depreciation(P, Salvage, Lifetime):   
    B = P
    S = Salvage
    N = Lifetime
                                                
    dt = lambda t: (B - S) / N # deprecation in year t
    bv = lambda t: (B - dt(t)*t) # total depreciation up to year t
    declinesstraight = [0] + [round(dt(x+1), 2) for x in range(N)]                 
    valuesstraight = [P] + [round(bv(x+1), 2) for x in range(N)]                
                                           
                                                
                                                
    calc_d = lambda: ((B - S) / N) / (B - S) * 2               
    D = calc_d()                                                     
    dt = lambda t: D * B * (1-D)**(t-1) # deprecation in year t      
    bv = lambda t: B * (1 - D) ** t # total depreciation up to year t
    declinesddb = [0] + [round(dt(x+1), 2) for x in range(N)]                 
    valuesddb = [P] + [round(bv(x+1), 2) for x in range(N)]  

    d = {
        "Book value Straightline": valuesstraight,
        "Book value DDB": valuesddb,
        "Declines Straightline": declinesstraight,
        "Declines Double Declining": declinesddb,
    }
    print("NOTE: if the book value is below the salvage value, it should instead be the salvage value. This also affects the depreciation amount")

    df = pd.DataFrame(data=d)

    ax = plt.gca()
    df.plot(kind='line', y="Book value Straightline", ax=ax)
    df.plot(kind='line', y="Book value DDB", ax=ax)
    plt.axhline(y=Salvage, label="Salvage Value")
    plt.legend()
    xa = ax.get_xaxis()
    xa.set_major_locator(MaxNLocator(integer=True))
    
    plt.show()
    

    return df
 
def depreciation_units_of_production_rate(P, Salvage, units):
    return (P - Salvage)/ units


def break_even_change_selling(Fixed_cost, Variable_cost_ratio, change):
    return Fixed_cost / (1- Variable_cost_ratio/(1 + change))

# cost includes installation and transportation
def CCA_with_50(P, d, year):
    if year == 1:
        return P * d/2

    return P * d * (1 - d/2) * (1 - d) ** (year - 2)

# undepreciated capital cost allowance i.e. original buying price - cca
def UCC_with_50(P, d, year):
    return P * (1 - d/2) * (1-d) ** (year - 1)

# after 1984 assume 4% cca
# land does not depreciate
def CCA_without_50(P, d, year):
    if year == 1:
        return P * d
    return P*d * (1-d) ** (year-1)

def UCC_without_50(P, d, year):
    return P * (1-d) ** year

def net_proceeds_with_50(P, d, year, tax_rate, sell_price):
    value = UCC_with_50(P, d, year)
    print(f"value at year {year}: {value}")
    taxable_loss = value - sell_price
    print(f"taxable loss: {taxable_loss}")
    tax_credit = tax_rate * taxable_loss
    print(f"tax credit: {tax_credit}")
    net_proceeds = sell_price + tax_credit
    print(f"net proceeds: {net_proceeds}")
    return net_proceeds

def tax(income, lower_provincial=None, higher_provincial=None, lower_federal=None, higher_federal=None):
    sum = 0

    if lower_provincial is not None and higher_provincial is not None:
        sum += min(400000, income) * lower_provincial + max((income - 400000), 0) * higher_provincial

    if lower_federal is not None and higher_federal is not None:
        sum += min(400000, income) * lower_federal + max((income - 400000), 0) * higher_federal

    print(f"after tax income: {income - sum}")
    return sum

def disposal_tax_effect(P, d, year, sell_price, tax):
    ucc = UCC_with_50(P, d, year)
    gain_in_sale = sell_price - ucc

    disposal_tax = tax * gain_in_sale

    return disposal_tax


def SOYD_depreciation(P, N, n, S):
    soyd = sum(range(1, N+1))
    return (N-n + 1)/soyd * (P-S)
    
def UP_depreciation(Units, Total, P, S):
    return Units/Total * (P-S)

# cost basis is the total cost to get the asset into usable condition
# includes initial cost and freight and installation
# if it's a trade in,
# book value = cost basis - total accumulated depreciation
# salvage value must be considered
# subtract the salvage - book value (unrecognized gains)
# also subtract trade in allowance

# declining balance
# d = 1/N * multiplier

