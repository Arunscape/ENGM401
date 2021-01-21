import numpy as np
import numpy_financial as npf
import sympy as sym
import pandas as pd
import matplotlib.pyplot as plt

# simple interest
def fv_simple(P, i, N) -> float:
    return P * (1 + i*N)
# compound interest
def fv_compound(P, i, N) -> float:
    return P * (1 + i)**N