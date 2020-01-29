import numpy as np
from numerical_methods import *
from time import perf_counter as timer

ref = np.sqrt(5)
x0 = 1
h = 0.01
tolerance = 0.01
R = root()

def f(x):
    return x**2 - 5

start = timer()
out = R.newton(f, x0, tolerance)
print( 'E_newton =', R.value - ref, '|| iterations= ', out[1] )
elapsed = timer() - start
print("Time =", elapsed, "\n")

start = timer()
out = R.secant(f, x0, tolerance)
print( 'E_secant =', R.value - ref, '|| iterations= ', out[1] )
elapsed = timer() - start
print("Time =", elapsed, "\n")
