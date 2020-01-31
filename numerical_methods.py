import numpy as np

class integral:
    def __init__(self):
        self.value = 0

    def trapezoid(self, f, lim, N):
        self.value = 0
        h = (lim[1]-lim[0])/N
        k = h/2
        x = lim[0] + h
        for i in range(1, N, 2):
            self.value += k*( f(x - h) + 2*f(x) + f(x + h))
            x += 2*h
        return self.value

    def simpson(self, f, lim, N):
        self.value = 0
        h = (lim[1]-lim[0])/N
        k = h/3
        x = lim[0] + h
        for i in range(1, N, 2):
            self.value += k*( f(x-h) + 4*f(x) + f(x+h) )
            x += 2*h
        return self.value

    def boole(self, f, lim, N):
        self.value = 0
        h = (lim[1]-lim[0])/N
        k = 2*h/45
        x = lim[0]
        for i in range(0, N, 4):
            self.value += k*( 7*f(x) + 32*f(x+h) + 12*f(x + 2*h) + 32*f(x + 3*h) + 7*f(x + 4*h) )
            x += 4*h
        return self.value


class derivative:
    def __init__(self):
        self.value = 0

    def fd(self, f, x, h):
        self.value = 0
        self.value = ( f(x+h) - f(x) )/h
        return self.value

    def bd(self, f, x, h):
        self.value = 0
        self.value = ( f(x) - f(x-h) )/h
        return self.value

    def threep(self, f, x, h):
        self.value = 0
        self.value = ( f(x+h) - f(x-h) )/(2*h)
        return self.value

    def fivep(self, f, x, h):
        self.value = 0
        self.value = ( f(x-2*h) - 8*f(x-h) + 8*f(x+h) -f(x+2*h) )/(12*h)
        return self.value


class root:
    def __init__(self):
        self.value = 0

    def newton(self, f, x0, tolerance):
        x_new = np.random.random()*17
        x = x0
        max_iter = 100
        k = 0
        while (abs(x_new - x) > tolerance) & (k <= max_iter):
            x = x_new
            x_new = x - ( f(x) / derivative().fivep(f, x, 0.001) )
            k += 1

        self.value = x_new
        return self.value, k

    def secant(self, f, x0, tolerance):
        x_new = np.random.random()*17
        x = x0
        x_old = 0
        max_iter = 100
        k = 0
        while (abs(x_new - x) > tolerance) & (k <= max_iter):
            x = x_new
            x_new = x - f(x) * ( (x-x_old) / (f(x)-f(x_old)) )
            x_old = x
            k += 1

        self.value = x_new
        return self.value, k

class ode:
    def __init__(self):
        self.value = 0

    def euler(self, f, bc, lim, N):
        self.value = 0
        h = (lim[1]-lim[0])/N
        x = lim[0]
        y = bc
        for i in range(0, N):
            y_new = y + h*f(x, y)
            x += h
            y = y_new

        self.value = y_new
        return self.value

    def taylor(self, f, dfdx, dfdy, bc, lim, N):
        # Where f(x, y) = dy/dx
        self.value = 0
        h = (lim[1]-lim[0])/N
        x = lim[0]
        y = bc
        k = 0.5*h**2
        for i in range(0, N):
            y_new = y + h*f(x, y) + k*(dfdx(x, y)+f(x, y)*dfdy(x, y))
            x += h
            y = y_new

        self.value = y_new
        return self.value

    def implicit(self, f, g, bc, lim, N):
        # Where f(x, y) = dy/dx
        self.value = 0
        h = (lim[1]-lim[0])/N
        x = lim[0]
        y = bc
        k = 0.5*h
        for i in range(0, N):
            y_new = (( 1+k*g(x) )/( 1-k*g(x+h) ))*y
            x += h
            y = y_new

        self.value = y_new
        return self.value

    def rk2(self, f, bc, lim, N):
        self.value = 0
        h = (lim[1]-lim[0])/N
        x = lim[0]
        y = bc
        for i in range(0, N):
            k = h*f(x, y)
            y_new = y + h*f(x+h/2, y+k/2)
            x += h
            y = y_new

        self.value = y_new
        return self.value


    def rk3(self, f, bc, lim, N):
        self.value = 0
        h = (lim[1]-lim[0])/N
        x = lim[0]
        y = bc
        for i in range(0, N):
            k1 = h*f(x, y)
            k2 = h*f(x+h/2, y+k1/2)
            k3 = h*f(x+h, y - k1+2*k2)
            y_new = y + 1/6*(k1 + 4*k2 + k3)
            x += h
            y = y_new

        self.value = y_new
        return self.value

    def rk4(self, f, bc, lim, N):
        self.value = 0
        h = (lim[1]-lim[0])/N
        x = lim[0]
        y = bc
        for i in range(0, N):
            k1 = h*f(x, y)
            k2 = h*f(x+h/2, y+k1/2)
            k3 = h*f(x+h/2, y+k2/2)
            k4 = h*f(x+h, y+k3)
            y_new = y + 1/6*(k1 + 2*k2 + 2*k3 + k4)
            x += h
            y = y_new

        self.value = y_new
        return self.value
