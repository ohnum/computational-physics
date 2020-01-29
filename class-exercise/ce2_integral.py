import numpy as np
from numerical_methods import *
from time import perf_counter as timer

ref = 1.718282
N = 400000 # partitions of x
limit = [0, 1]
I = integral()


def f(x):
    return np.exp(x)


print('N =', N)
start = timer()
print( 'E_trapezoid =', I.trapezoid(f, limit, N) - ref )
elapsed = timer() - start
print("Time =", elapsed, "\n")

start = timer()
print( 'E_simpson =', I.simpson(f, limit, N) - ref )
elapsed = timer() - start
print("Time =", elapsed, "\n")

start = timer()
print( 'E_boole =', I.boole(f, limit, N) - ref )
elapsed = timer() - start
print("Time =", elapsed, "\n")
