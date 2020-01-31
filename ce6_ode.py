import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import *
from time import perf_counter as timer

# Define f(x, y) = dy/dx, and its partial derivatives
def f(x, y):
    return -x*y

def dfdx(x, y):
    return -y

def dfdy(x, y):
    return -x

def g(x):
    return -x

# Setup the initial conditions and the limits x in [0,3] or x in [0,1]
ode = ode()
x = 1.0
ref = np.exp(-x**2/2)
limit = [0, x]
y0 = 1.0

N = 2000 # partitions of x
print('N =', N)
solution = np.zeros(N)

#Euler method
print('ODE_euler')
start = timer()
for i in range(N):
    solution[i] = ode.euler(f, y0, limit, i+1)
plt.plot(np.arange(N), np.log10(abs(solution-ref)), label="Euler")
elapsed = timer() - start
print("Time =", elapsed, "\n")


# Taylor method
print('ODE_taylor')
start = timer()
for i in range(N):
    solution[i] = ode.taylor(f, dfdx, dfdy, y0, limit, i+1)
plt.plot(np.arange(N), np.log10(abs(solution-ref)), label="Taylor")
elapsed = timer() - start
print("Time =", elapsed, "\n")

# Implicit method
print('ODE_implicit')
start = timer()
for i in range(N):
    solution[i] = ode.implicit(f, g, y0, limit, i+1)
plt.plot(np.arange(N), np.log10(abs(solution-ref)), label="Implicit")
elapsed = timer() - start
print("Time =", elapsed, "\n")

#RK2
print('ODE_rk2')
start = timer()
for i in range(N):
    solution[i] = ode.rk2(f, y0, limit, i+1)
plt.plot(np.arange(N), np.log10(abs(solution-ref)), label="RK2")
elapsed = timer() - start
print("Time =", elapsed, "\n")


# RK3
print('ODE_rk3')
start = timer()
for i in range(N):
    solution[i] = ode.rk3(f, y0, limit, i+1)
plt.plot(np.arange(N), np.log10(abs(solution-ref)), label="RK3")
elapsed = timer() - start
print("Time =", elapsed, "\n")

# RK4
print('ODE_rk4')
start = timer()
for i in range(N):
    solution[i] = ode.rk4(f, y0, limit, i+1)
plt.plot(np.arange(N), np.log10(abs(solution-ref)), label="RK4")
elapsed = timer() - start
print("Time =", elapsed, "\n")

plt.legend()
plt.show()
