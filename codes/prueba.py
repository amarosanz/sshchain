import numpy as np 
from scipy.optimize import bisect

M=120000
D=0.7
def f(x):
    return np.tan(x*(M+1))-D*np.sin(x)/(1+D*np.cos(x))
    
roots=[]
    
a_prev = np.pi*(0.5)/(M+1)


for i in range(1, M+1):
    a = np.pi*(i+0.5)/(M+1)
    while f(a_prev)>0 :
        a_prev = a_prev+1e-7
    while f(a)<0 : 
        a = a-1e-7
 
    print("a=", a_prev)
    print("b=", a)
    print("f(a)=", f(a_prev))
    print("f(b)=",f(a))
    sol = bisect(f, a_prev, a, xtol=1e-15)
    print("solucion",sol)
    roots.append(sol)
    a_prev = a
   
print(roots) 
print(len(roots))
    
    
    

