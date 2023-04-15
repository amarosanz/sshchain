import numpy as np
import matplotlib.pyplot as plt
M=10
D=0.6
def f(x,M,D):
    return np.tan(x*M)-D*np.sin(x)/(1+D*np.cos(x))

def df(x,M,D):
    return M/(np.cos(M*x)**2) - D**2*np.sin(x)**2/(D*np.cos(x)+1)**2 - D*np.cos(x)/(D*np.cos(x)+1)


def newton(f, df, x0, eps, max_iter):
    x = x0
    for i in range(max_iter):
        fx = f(x,M,D)
        if abs(fx) < eps:
            return x
        dfx = df(x,M,D)
        if dfx == 0:
            return None
        x = x - fx/dfx
    return None

a = 0
b = np.pi

roots = []

n = 1000
dx = (b - a)/n
for i in range(n):
    x0 = a + i*dx
    x1 = a + (i+1)*dx
    root = newton(f, df, x0, 1e-6, 1000)
    if root and root >= x0 and root <= x1:
        roots.append(root)

print("Raíces encontradas:", roots)

x = np.linspace(a, b, 1000)
y = f(x,M,D)

plt.plot(x, y)
for root in roots:
    plt.plot(root, f(root,M,D), 'ro')
plt.show()

