#import numpy as np 
#from scipy.optimize import bisect
#import matplotlib.pyplot as plt

#D=0.7
#def f(x):
#    return np.tan(x*(M+1))-D*np.sin(x)/(1+D*np.cos(x))
#    
#roots=[]
#    
#a_prev = np.pi*(0.5)/(M+1)


#for i in range(1, M+1):
#    a = np.pi*(i+0.5)/(M+1)
#    while f(a_prev)>0 :
#        a_prev = a_prev+1e-7
#    while f(a)<0 : 
#        a = a-1e-7
# 
#    print("a=", a_prev)
#    print("b=", a)
#    print("f(a)=", f(a_prev))
#    print("f(b)=",f(a))
#    sol = bisect(f, a_prev, a, xtol=1e-15)
#    print("solucion",sol)
#    roots.append(sol)
#    a_prev = a
#   
#print(roots) 
#print(len(roots))
#k1 = np.linspace(0.1,np.pi-0.1, 1000)
#A1 = 1/np.sqrt(M+0.5*(1-np.sin((2*M+1)*k1)/np.sin(k1)))

#plt.plot(k1,A1**2)
#plt.show()




"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

def fq(x, D, M):
    return D * np.sinh(x * M) - np.sinh(x * (M + 1))

M = 5
Dc = M/(M+1)
print("Dc = ", Dc)  
D_values = np.linspace(1/Dc+0.001, 5, 1000)  # Valores de D para los cuales se calcularán las raíces
roots = []  # Lista para almacenar las raíces encontradas

# Calcular las raíces para cada valor de D
for D in D_values:
    print("D = ", D)
    print("\nMc = ", D/(D-1))
    # Definir la función auxiliar que toma solo el argumento x
    def fq_aux(x):
        return fq(x, D, M)
    
    # Encontrar la raíz utilizando bisect
    qlim = np.log(D)
    q = bisect(fq_aux, 0 + 1e-7, qlim+1e-7, xtol=1e-16)
    roots.append(q/qlim)

# Representar las soluciones en función de D
plt.plot(D_values, roots)
plt.xlabel('D')
plt.ylabel('q/qlim')
plt.title('Roots of fq')
plt.grid(True)
plt.show()
"""

#import numpy as np
#import numpy.linalg as linalg
#from scipy.linalg import block_diag
#from scipy.optimize import bisect
#import math
#import time

#t1 = 1/1.2 #integral de salto intra-cell   
#t2 = 1.2 #integral de salto inter-cell
#D=t2/t1
#N = 20 #número de átomos de la cadena 
#M = N//2 #número de celdas unidad de la cadena
#Mc = 1/(D-1) #número de celdas unidad crítico de la cadena
#par = N % 2 == 0

#print("\nCADENA {}DE N = {} ÁTOMOS".format("PAR " if par else "IMPAR ", N))
#print("\nMc =", Mc)
#print("t1 =", t1)
#print("t2 =", t2)
#print("D = t2/t1 =", D)

#def hfinal(t1,t2,N):
#    start_time = time.time() # Comienzo de la medición de tiempo 
#    matriz_diag = np.diag(np.zeros(N))
#    matriz_superior_inferior = np.zeros((N,N))
#    for i in range(N-1):
#        if i % 2 == 0:
#            matriz_superior_inferior[i,i+1] = t2
#        else:
#            matriz_superior_inferior[i,i+1] = t1
#        
#    for i in range(N-1):
#        if i % 2 == 0:
#            matriz_superior_inferior[i+1,i] = t2
#        else:
#            matriz_superior_inferior[i+1,i] = t1
#    matriz_final = matriz_diag + matriz_superior_inferior
#    
#    elapsed_time = time.time() - start_time # Tiempo de ejecución
#    print(f"\nTiempo de ejecución de la función {hfinal.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución
#    
#    return matriz_final 


#def eigen(t1,t2,N):
#    start_time = time.time() # Comienzo de la medición de tiempo 
#    h=hfinal(t1,t2,N)
#    values, states = np.linalg.eigh(h)
#    idx = values.argsort()[::-1]   
#    values = values[idx]
#    states = np.transpose(states[:,idx])
#    elapsed_time = time.time() - start_time # Tiempo de ejecución
#    print(f"Tiempo de ejecución de la función {eigen.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución
#    
#    return values,states


#def ks(N):
#    h = hfinal(t1, t2, N)
#    values, states = np.linalg.eigh(h)
#    k = []
#    for v in values.tolist():
#        pos = values.tolist().index(v)
#        x = ((v / t1) ** 2 - 1 - D ** 2) / (2 * D)
#        kv = np.arccos(x)
#        k.append(kv)
#    print(k)
#    print(len(k))

#print(hfinal(t1,t2,N))    
#ks(N)


import math
import numpy as np
import matplotlib.pyplot as plt

def function(delta, q):
    return math.sqrt(1 + delta**2 - 2*delta*math.cosh(q))

delta = 1.2
q_values = np.linspace(0, math.log(delta), 100)  # Genera 100 valores equidistantes entre 0 y ln(Delta)
results = [function(delta, q) for q in q_values]

plt.plot(q_values, results)
plt.xlabel('q')
plt.ylabel('f(q)')
plt.title('Representación de la función sqrt(1 + ∆^2 − 2∆ cosh(q))')
plt.grid(True)
plt.show()















#qs = np.linspace(-qlim,qlim,1000)
#plt.plot(qs,fq(qs,D,M))
#plt.show()




    

