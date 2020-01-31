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
x = 1
ref = np.exp(-0**2/2)
y0 = 1
eps = 10**-10
limit = [eps, x]

N = 1000 # partitions of x
print('N =', N)
solution_backward = np.zeros(N)

#Euler method
print('ODE_euler')
start = timer()
for i in range(N):
    limit = [0, x] # forward limit
    y0 = 1 # bc forward
    solution_forward = ode.euler(f, y0, limit, i+1) # forward

    limit = [x, 0] # backward limit
    y0 = solution_forward # new initial value (then one we previously calculated)
    solution_backward[i] = ode.euler(f, y0, limit, i+1) # backward

plt.plot(np.arange(N), np.log10(abs(solution_backward-ref)), label="Euler")
elapsed = timer() - start
#print(solution_backward[N-10:N])
#print(solution_backward)
print("Time =", elapsed, "\n")


# Taylor method
print('ODE_taylor')
start = timer()
for i in range(N):
    limit = [eps, x] # forward limit
    y0 = 1 # bc forward
    solution_forward = ode.taylor(f, dfdx, dfdy, y0, limit, i+1) # forward

    limit = [x, eps] # backward limit
    y0 = solution_forward # new initial value (then one we previously calculated)
    solution_backward[i] = ode.taylor(f, dfdx, dfdy, y0, limit, i+1) # backward

plt.plot(np.arange(N), np.log10(abs(solution_backward-ref)), label="Taylor")
elapsed = timer() - start
#print(solution_backward[N-10:N])
#print(solution_backward)
print("Time =", elapsed, "\n")


# Implicit method
print('ODE_implicit')
start = timer()
for i in range(N):
    limit = [eps, x] # forward limit
    y0 = 1 # bc forward
    solution_forward = ode.implicit(f, g, y0, limit, i+1) # forward

    limit = [x, eps] # backward limit
    y0 = solution_forward # new initial value (then one we previously calculated)
    solution_backward[i] = ode.implicit(f, g, y0, limit, i+1) # backward

plt.plot(np.arange(N), np.log10(abs(solution_backward-ref)), label="Implicit")
elapsed = timer() - start
#print(solution_backward[N-10:N])
#print(solution_backward)
print("Time =", elapsed, "\n")

plt.legend()
plt.show()
