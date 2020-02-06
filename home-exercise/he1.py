import sys
sys.path.append("../class-exercise/")
import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import *
from time import perf_counter as timer

def dpdt(dummy, y):
    return -4*np.pi**2*y

def dydt(dummy, p):
    return p

def reference(t):
    # c1 and c2 are found analytically by setting y(0) = 0 and p(0) = 1
    c1 = 1/(2*np.pi)
    c2 = 0
    return c1*np.sin(2*np.pi*t) + c2*np.cos(2*np.pi*t)

# def test(**kwargs):
#     print(kwargs.keys())
#     print(kwargs.values())
#
# print(test(x=1, y=2))


N = 50000 # partitions of t
t = 5.0 #end value of t
h = t/N # step size
ode2 = ode2() # creating an object of the class ode2. From numerical_methods.py
solution = np.zeros([N, 2])

# RK2
y = 0
p = 1
for i in range(0, N):
    y, p = ode2.rk2(dydt, dpdt, [y, p], [0, h], 1)
    solution[i, 0] = y
    solution[i, 1] = p

# RK3
y = 0
p = 1
for i in range(0, N):
    y, p = ode2.rk3(dydt, dpdt, [y, p], [0, h], 1)
    solution[i, 0] = y
    solution[i, 1] = p

# RK4
print('1. RK4 solution to d^2y/dt^2 = -4pi^2*y.')
print('N = ', N)
print('t_end = ', t)
print('h = ', t/N)
y = 0
p = 1
for i in range(0, N):
    y, p = ode2.rk4(dydt, dpdt, [y, p], [0, h], 1)
    solution[i, 0] = y
    solution[i, 1] = p

x = np.linspace(0, t, N)
plt.plot(x, solution[:, 0], '.', label="RK4, displacement")
plt.plot(x, solution[:, 1], '-', label="RK4, momentum")
plt.plot(x, reference(x), '-', label="Reference, displacement")
plt.xlabel("Time t")
plt.ylabel("Displacement y, momentum p")
plt.legend(loc="lower right")
print('Done.')


print('2. Deviation from reference solution 1/2pi*sin(2pi*t)')
start = 100 # start value of N
N = 1000 # partitions of t
stop = N + start # stop value of N
times = [0.5, 1.0, 1.5, 2.0] # some different end values of t
ref = reference(t) # reference solutions or the different times
y = 0 #y(0)
p = 1 #p(0)
solution = np.zeros([N, 2])
plt.figure()
print('Calculating solutions for times:')
print('times =', times, ',')
print('for different values of N between:')
print(str(start) + ' <= N <= '+ str(stop))
for t in times:
    for i in range(0, N):
        solution[i, :] = ode2.rk4(dydt, dpdt, [y, p], [0, t], i+start)

    plt.plot(np.arange(start, stop), np.log10(abs(solution[:, 0] - ref)), label="RK4, t="+str(t))

plt.xlabel("Partitions N on t (h=(b-a)/N)")
plt.ylabel("log10(Error)")
print('Done.')

plt.legend(loc="upper right")
plt.show()
print('Done.')
