import numpy as np
from numerical_methods import *
from time import perf_counter as timer

ref = np.cos(1)
x = 1
h = 0.1
D = derivative()

def f(x):
    return np.sin(x)


start = timer()
print( 'E_fdif =', D.fd(f, x, h) - ref )
elapsed = timer() - start
print("Time =", elapsed, "\n")

start = timer()
print( 'E_bdif =', D.bd(f, x, h) - ref )
elapsed = timer() - start
print("Time =", elapsed, "\n")

start = timer()
print( 'E_3p =', D.threep(f, x, h) - ref )
elapsed = timer() - start
print("Time =", elapsed, "\n")

start = timer()
print( 'E_5p =', D.fivep(f, x, h) - ref )
elapsed = timer() - start
print("Time =", elapsed, "\n")
