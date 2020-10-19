import numpy as np

eps = 1e-17

class utility:
    def __init__(self):
        self.value = 0

    def recursive_scale(self, factor, n, stop, list):
        if n > stop:
            return list
        else:
            list.append(n)
            return self.recursive_scale(factor, factor*n, stop, list)

class integral:
    def __init__(self):
        self.value = 0

    def trapezoid(self, f, lim, h):
        self.value = 0
        k = h/2
        x = lim[0]
        start = int(lim[0]/h)
        stop = int(lim[1]/h)
        for i in range(start+1, stop, 2):
            self.value += k*( f(x - h) + 2*f(x) + f(x + h))
            x += 2*h
        return self.value

    def simpson(self, f, lim, h, args=()):
        self.value = 0
        k = h/3
        x = lim[0]
        start = int(lim[0]/h)
        stop = int(lim[1]/h)
        for i in range(start+1, stop, 2):
            self.value += k*( f(x-h, args[0], args[1]) + 4*f(x, args[0], args[1]) + f(x+h, args[0], args[1]) )
            x += 2*h
        return self.value

    def boole(self, f, lim, h):
        self.value = 0
        k = 2*h/45
        x = lim[0]
        start = int(lim[0]/h)
        stop = int(lim[1]/h)
        for i in range(start, stop, 4):
            self.value += k*( 7*f(x) + 32*f(x+h) + 12*f(x + 2*h) + 32*f(x + 3*h) + 7*f(x + 4*h) )
            x += 4*h
        return self.value

    def legendre(self, n, x):
        p = np.zeros(n+1)
        q = np.zeros(n+1)
        p[0] = 1
        q[0] = 0
        p[1] = x
        q[1] = 1
        for i in range(2,n+1):
            p[i] = ((2*(i-1)+1)*x*p[i-1] - (i-1)*p[i-2])/(i)
            q[i] = (-i*x*p[i] + i*p[i-1])/(1-x*x)
        return p[n],q[n]

    def find_root(self, n, f, x0, xf):
        delta = 10**-12
        while(abs(x0 - xf) > delta):
            x_new = 0.5*(x0 + xf)
            if np.sign(f(n, x_new)[0]) == np.sign(f(n, x0)[0]):
                x0 = x_new
            else:
                xf = x_new
        avg = 0.5*(xf+x0)
        if (abs(avg)<delta):
            return 0
        return avg

    def roots(self, n, N=-1):
        if(N==-1):
            N=4*n+1
        count = 0
        Roots = []
        x = np.linspace(-1,1, N+1)
        for i in range(N):
            f0 = self.legendre(n, x[i])[0]
            ff = self.legendre(n, x[i+1])[0]
            if (np.sign(f0)!=np.sign(ff)):
                count+=1
                Roots.append(self.find_root(n, self.legendre, x[i], x[i+1]))
        if(n == len(Roots)):
            return Roots

        return self.roots(n, (N-1)*2+1)

    def weights(self, n, x):
        p = self.legendre(n, x)[1]
        wg = 2/((1-x*x)*(p**2))
        return wg

    def quadrature(self, f, root, lim=[-1,1], args=()):
        N = len(root)
        x0 = lim[0]
        xf = lim[1]
        ws = []
        sum = 0
        for i in range(N):
            ws.append(self.weights(N, root[i]))
            x = x0 + (root[i]+1) * (xf-x0)/2
            sum += ws[i] * f(x, args[0], args[1])
        ratio = (xf-x0)/(1-(-1)) #result from stretching the interval
        return sum*ratio


class derivative:
    def __init__(self):
        self.value = 0

    def fd(self, f, x, h, **param):
        self.value = 0
        self.value = ( f(x=x+h, **param) - f(x=x, **param) )/h
        return self.value

    def bd(self, f, x, h):
        self.value = 0
        self.value = ( f(x) - f(x-h) )/h
        return self.value

    def threep(self, f, x, h, **param):
        self.value = 0
        self.value = ( f(x=x+h, **param) - f(x=x-h, **param) )/(2*h)
        return self.value

    def fivep(self, f, x, h, **param):
        self.value = 0
        self.value = ( f(x=x-2*h, **param) - 8*f(x=x-h, **param) + 8*f(x=x+h, **param) - f(x=x+2*h, **param) )/(12*h)
        return self.value


class root:
    def __init__(self):
        self.value = 0

    def bisection(self, f, x_left, x_right, tolerance=1e-5):
        while(abs(x_left - x_right) > tolerance):
            x_center = 0.5*(x_left + x_right)
            f_center = f(x_center)
            f_left = f(x_left)
            if np.sign(f_center) == np.sign(f_left):
                x_left = x_center
            else:
                x_right = x_center
        return 0.5*(x_left + x_right)

    def newton(self, f, x0, tolerance=1e-5):
        x_new = x0/10
        x = x0
        max_iter = 100
        k = 0
        while (abs(x_new - x) > tolerance) & (k <= max_iter):
            x = x_new
            x_new = x - ( f(x) / derivative().fivep(f, x, tolerance) )
            k += 1

        self.value = x_new
        return self.value, k

    def secant(self, f, x0, tolerance=1e-5):
        x_new = x0/10
        x = x0
        x_old = x_new/2
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
    # f(x, y) = dy/dx
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


class ode2:
    # f(x, y) = dy/dx
    def __init__(self):
        self.value = [0,0]

    def rk2(self, f, f_prime, bc, lim, N):
        self.value = [0,0]
        h = (lim[1]-lim[0])/N
        x = lim[0]
        y = bc[0]
        z = bc[1]
        for i in range(0, N):
            k = h*f_prime(x, y)
            z_new = z + h*f_prime(x+h/2, y+k/2)

            k = h*f(x, z)
            y_new = y + h*f(x+h/2, z+k/2)

            x += h
            y = y_new
            z = z_new

        self.value[0] = y
        self.value[1] = z
        return y, z


    def rk3(self, f, f_prime, bc, lim, N):
        self.value = [0,0]
        h = (lim[1]-lim[0])/N
        x = lim[0]
        y = bc[0]
        z = bc[1]
        for i in range(0, N):
            k1 = h*f_prime(x, y)
            k2 = h*f_prime(x+h/2, y+k1/2)
            k3 = h*f_prime(x+h, y-k1+2*k2)
            z_new = z + 1/6*(k1 + 4*k2 + k3)

            k1 = h*f(x, z)
            k2 = h*f(x+h/2, z+k1/2)
            k3 = h*f(x+h, z-k1+2*k2)
            y_new = y + 1/6*(k1 + 4*k2 + k3)

            x += h
            y = y_new
            z = z_new

        self.value[0] = y
        self.value[1] = z
        return y, z

    def rk4(self, f, f_prime, bc, lim, N):
        self.value = [0,0]
        h = (lim[1]-lim[0])/N
        x = lim[0]
        y = bc[0]
        z = bc[1]
        for i in range(0, N):
            k1 = h*f_prime(x, y)
            k2 = h*f_prime(x+h/2, y+k1/2)
            k3 = h*f_prime(x+h/2, y+k2/2)
            k4 = h*f_prime(x+h, y+k3)
            z_new = z + 1/6*(k1 + 2*k2 + 2*k3 + k4)

            k1 = h*f(x, z)
            k2 = h*f(x+h/2, z+k1/2)
            k3 = h*f(x+h/2, z+k2/2)
            k4 = h*f(x+h, z+k3)
            y_new = y + 1/6*(k1 + 2*k2 + 2*k3 + k4)

            x += h
            y = y_new
            z = z_new

        self.value[0] = y
        self.value[1] = z
        return y, z
