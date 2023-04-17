import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

######################################################################
######################################################################
#Programa para calcular las soluciones (vectores de onda) de bulk. La fase trivial D>1 debería dar M soluciones, la fase topólogica M-1 si M>D/(1-D), M si M<D/(1-D)
M = 150
D = 0.3/0.7
Mc = D/(1-D)


if D<1:
    expected_phase = "Topological"
    if M>Mc: 
        number_bsol = M-1
    else: 
        number_bsol = M
else: 
    expected_phase = "Trivial" 
    number_bsol = M
     

def f(x,M,D):
    return np.tan(x*(M))+D*np.sin(x)/(1+D*np.cos(x))

def df(x,M,D):
    return M*(1/np.cos(M*x))**2 + (D**2*(np.sin(x)*np.sin(x)))/((D*np.cos(x)+1)*(D*np.cos(x)+1)) + D*np.cos(x)/(D*np.cos(x)+1)



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
b = np.pi-10**(-6)

roots = []

n = 1000 #Aumentar para cadenas más grandes (número de subintervalos en los que buscamos la solución)
dx = (b - a)/n
for i in range(n):
    x0 = a + i*dx
    x1 = a + (i+1)*dx
    root = newton(f, df, x0, 1e-6, 1000)
    if root and root >= x0 and root <= x1:
        roots.append(root)

#print("Raíces encontradas:", roots)
#for i in roots: print("solución ", roots.index(i)+1," x=",i, "; f(x,M,D):", f(i,M,D))
print("M = ", M)
print("D = ", D)
print("Mc = ",Mc)



table = []
for i in roots:
    sublist=[roots.index(i)+1,i,f(i,M,D)]
    table.append(sublist)

print(tabulate(table, headers=["Nº", "Solution", "Distance to 0"]))
print("Número de soluciones encontradas =", len(roots))
x = np.linspace(a, b, 1000)
y = f(x,M,D)

plt.plot(x, y)
for root in roots:
    plt.plot(root, f(root,M,D), 'ro')
#plt.show()

