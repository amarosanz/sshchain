######################################################################
######################################################################
#Programa para calcular las autoenergías y los autoestados de una cadena SSH finita de N celdas unidad 
import sys, os
import numpy as np
import numpy.linalg as linalg
from scipy.linalg import block_diag
from scipy.optimize import bisect
from scipy.optimize import newton
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import math
import time
import random
import itertools


######################################################################
######################################################################

#Definición constantes
t1=0.6 #integral de salto intra-cell   
t2=0.7 #integral de salto inter-cell
d=t2/t1
N = 40 #número de átomos de la cadena 
Mc = 1/(d-1)
print("t1 = ", t1)
print("t2 = ", t2)
print("D = ", d)
if N%2 == 0:
    print("M = ", N//2)
else:
    print("Cadena impar de N = ",N)
print("Mc =", Mc)

parent_dir = os.path.dirname(os.getcwd())
if (not os.path.exists(os.path.join(parent_dir, "plots_ssh"))):
    os.mkdir(os.path.join(parent_dir, "plots_ssh"))
ruta = os.path.join(parent_dir, "plots_ssh")
ruta_N= ruta+"/"+str(N)+"chain"

######################################################################
######################################################################

#Construcción del hamiltoniano tridiagonal por bloques
def tridiag(c, u, d, N):
    start_time = time.time() # Comienzo de la medición de tiempo 
    cc = block_diag(*([c]*N))
    shift = c.shape[1]
    uu = block_diag(*([u]*N)) 
    uu = np.hstack((np.zeros((uu.shape[0], shift)), uu[:,:-shift]))
    dd = block_diag(*([d]*N)) 
    dd = np.hstack((dd[:,shift:],np.zeros((uu.shape[0], shift))))
    
    elapsed_time = time.time() - start_time # Tiempo de ejecución
    print(f"Tiempo de ejecución de la función {tridiag.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución

    return cc+uu+dd
    
def tridiagfinal(t1,t2,N):
    start_time = time.time() # Comienzo de la medición de tiempo 
    H0 = np.array([[0, t1], [t1, 0]])
    H1 = np.array([[0, 0], [t2, 0]])
    HM1 = np.array([[0, t2], [0, 0]])
    H = tridiag(H0,H1,HM1,N//2)
#    print(H)
    elapsed_time = time.time() - start_time # Tiempo de ejecución
    print(f"Tiempo de ejecución de la función {tridiagfinal.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución
    
    return H


    

                                   


#Matriz hamiltoniana de dimension impar 

def h_impar(t2,t1,N):
    start_time = time.time() # Comienzo de la medición de tiempo 
    matriz_diag = np.diag(np.zeros(N))
    matriz_superior_inferior = np.zeros((N,N))
    for i in range(N-1):
        if i % 2 == 0:
            matriz_superior_inferior[i,i+1] = t2
        else:
            matriz_superior_inferior[i,i+1] = t1
        
    for i in range(N-1):
        if i % 2 == 0:
            matriz_superior_inferior[i+1,i] = t2
        else:
            matriz_superior_inferior[i+1,i] = t1
    matriz_final = matriz_diag + matriz_superior_inferior
    
    elapsed_time = time.time() - start_time # Tiempo de ejecución
    print(f"Tiempo de ejecución de la función {h_impar.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución
    
    return matriz_final 

######################################################################
######################################################################

#CÁLCULO EIGENVALUES Y EIGENSTATES

#EIGENENERGIES
def dospar(t1,t2,N):
    start_time = time.time() # Comienzo de la medición de tiempo 
    Ha = tridiagfinal(t1,t2,N)
    Hb = tridiagfinal(t2,t1,N)
    eigenvaluesa, eigenvectorsa = np.linalg.eigh(Ha)
    eigenvaluesb, eigenvectorsb = np.linalg.eigh(Hb)
    idx = eigenvaluesa.argsort()[::-1]   
    eigenvaluesa = eigenvaluesa[idx]
    eigenvectorsa = eigenvectorsa[:,idx]
    eigenvaluesb = eigenvaluesb[idx]
    eigenvectorsb = eigenvectorsb[:,idx]
    
    elapsed_time = time.time() - start_time # Tiempo de ejecución
    print(f"Tiempo de ejecución de la función {dospar.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución
    
    return eigenvaluesa,eigenvectorsa,eigenvaluesb,eigenvectorsb
    
def impar(t1,t2,N):
    start_time = time.time() # Comienzo de la medición de tiempo 
    H_imp=h_impar(t2,t1,N)
    eigenvaluesimp, eigenvectorsimp = np.linalg.eigh(H_imp)
    idx = eigenvaluesimp.argsort()[::-1]   
    eigenvaluesimp = eigenvaluesimp[idx]
    eigenvectorsimp = eigenvectorsimp[:,idx]
    
    elapsed_time = time.time() - start_time # Tiempo de ejecución
    print(f"Tiempo de ejecución de la función {impar.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución
    
    return eigenvaluesimp,eigenvectorsimp


#EIGENSTATES
def eigenstates(t1,t2,N):
    start_time = time.time() # Comienzo de la medición de tiempo 
    if N%2==0:
        _, eigenvectors, _, _ = dospar(t1,t2,N)
    else: 
        _, eigenvectors = impar(t1,t2,N)
    
    eigenvectors=np.transpose(eigenvectors)  
#    print("eigenvectors=",eigenvectors)

    pesos_a= []
    pesos_b= []
   
    for sublista in eigenvectors:
        prov_a = []
        prov_b = []
        
        for i, elemento in enumerate(sublista):
            if i % 2 == 0:
                prov_a.append(elemento)
            else:
                prov_b.append(elemento)
        
        pesos_a.append(prov_a)
        pesos_b.append(prov_b)
    

    
    elapsed_time = time.time() - start_time # Tiempo de ejecución
    print(f"Tiempo de ejecución de la función {eigenstates.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución
    return pesos_a, pesos_b 
    

#######################################################################
#######################################################################

#PLOTS 

#EIGENENERGIES
def plot_energias(t1,t2,N):
    start_time = time.time() # Comienzo de la medición de tiempo 
    plt.figure()
    plt.ylabel("Eigenenergies")
    plt.xlabel("Atom index")
    if N%2 ==0: 
        x = list(range(1,N+1))
        eigenvaluesa, _, eigenvaluesb, _ = dospar(t1,t2,N)
#        plt.scatter(x,sorted(eigenvaluesb), label="$t_1$="+str(t2)+", $t_2$="+str(t1),color="red", s=7)
        plt.scatter(x,sorted(eigenvaluesa), label="$t_1$="+str(t1)+", $t_2$="+str(t2), s=17, color="RED")
        plt.title(str(N)+"-atom chain eigenvalues")
        plt.grid(visible=True,lw=0.5)
        plt.legend(fontsize=13)
        if (not os.path.exists(ruta_N)):
            os.mkdir(ruta_N)
        outputp=ruta_N+"/"+"eigenvalues.png"
#        plt.show()
        plt.savefig(outputp)
            
    else:
        x_imp = list(range(1,N+1))
        eigenvaluesimp, _ = impar(t1,t2,N)
        plt.scatter(x_imp,sorted(eigenvaluesimp), label="$t_1$="+str(t1)+", $t_2$="+str(t2), s=17, color="red")
        plt.title(str(N)+"-atom chain eigenvalues")
        plt.grid(visible=True,lw=0.5)
        plt.legend(fontsize=13)
        if (not os.path.exists(ruta_N)):
            os.mkdir(ruta_N)
        outputp=ruta_N+"/"+"eigenvalues.png"
        plt.savefig(outputp)

         
        
    
    elapsed_time = time.time() - start_time # Tiempo de ejecución
    print(f"Tiempo de ejecución de la función {plot_energias.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución


    
#EIGENSTATES
        
def plot_eigenstates(t1,t2,N):
    start_time = time.time() # Comienzo de la medición de tiempo 
    
    X_par=[i for i in range(2, N+1,2)]
    X_impar=[i for i in range(1,N+1,2)]  
      
    for j in range(0,N):
        pesos_a,pesos_b=eigenstates(t1,t2,N)
        plt.figure()
#        print("Pesos átomos A",pesos_a)
#        print("Pesos átomos B",pesos_b)
        plt.bar(X_par,pesos_b[j],color="red", edgecolor="black",label="B atoms")
        plt.bar(X_impar,pesos_a[j],color="blue", edgecolor="black", label="A atoms")
        plt.title("Eigenstate number " + str(j+1) + " for $t_1$=" + str(t1) + ", $t_2$=" + str(t2))

        plt.legend()
        plt.xlabel("Atom index")
        plt.ylabel("Eigenstate weight")
        if (not os.path.exists(ruta_N)):
            os.mkdir(ruta_N)
        print(j)
        outputplot=ruta_N+"/"+"Eigenstate"+str(j+1)+".png"
        plt.savefig(outputplot)
  
    j+=1
    elapsed_time = time.time() - start_time # Tiempo de ejecución
    print(f"Tiempo de ejecución de la función {plot_eigenstates.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución

    


#######################################################################
#######################################################################

##EDGE STATES WEIGHT DECAY AS A FUNCTION OF N (NO FUNCIONA)

def estatedecay(t1,t2):
    Nlist=[i for i in range(14,300,2)]
    eps=(np.log(t2/t1))**(-1)
    print(eps)
    q=[]
    for i in Nlist:
        pesos_a, _=eigenstates(t1,t2,i)
        q.append(np.log(abs(pesos_a[int(i/2)][-1])/abs(pesos_a[int(i/2)][0])))
 

    plt.plot(Nlist,q,label="Pesos numéricos", color="green")
    Nlistcont=np.linspace(14,300,500)
    print("nl",Nlistcont)
    plt.plot(Nlistcont,(-1*(Nlistcont-1)/eps),label="Teoría",color="red")
    plt.legend()
    



#######################################################################
#######################################################################

# Definir la ecuación a resolver

def find_roots(M,D):
    a_prev = np.pi*(0.5)/(M+1)
    roots=[]
    #Ecuación para edge states    
    def fq(x):
        return np.tanh(x*(M+1))-np.sinh(x)/(np.cosh(x)-1/D)

       
    #Ecuación para bulk states
    def f(x):
        return np.tan(x*(M+1))-D*np.sin(x)/(1+D*np.cos(x))

    for i in range(1, M+2):
        a = np.pi*(i+0.5)/(M+1)
        while f(a_prev)>0 :
            a_prev = a_prev+1e-8
        if M<Mc and D>1 and i==M+1: 
            a = np.pi 
        while f(a)<0 : 
            a = a-1e-8
        if a>np.pi:
            break
#        print("a",a_prev)
#        print("b",a)
#        print("f(b)", f(a))
        sol = bisect(f, a_prev, a, xtol=1e-14)
#        print("Solucion", sol)
        if abs(f(sol))<1e-2:
#            print("sol ",sol)
            roots.append(sol)


        a_prev = a 
    
    bulk = roots
    if D>1 and M>Mc:    
        qsol = bisect(fq,0+10**(-8),np.pi,xtol=1e-20)
        if abs(fq(qsol))<1e-4:
            roots.append(qsol) #El último elemento de la lista roots es la solución de edge
        print("QSOL",qsol)
        bulk = roots[:-1]

    
    table = []
    th_eigenenergies = []
    
   

    for i in roots:
        if i == roots[-1] and len(bulk)==M-1:
            dif=fq(i)
            print("dif",dif)
            arg = 1+D**2-2*abs(D)*np.cosh(i, dtype = np.float128)
            print("arg",arg)
            e = np.sqrt(arg)
           
        else: 
            dif=f(i)
            e = np.sqrt(1+D**2+2*D*np.cos(i))
            
        sublist=[roots.index(i)+1,i,dif]
        table.append(sublist)
        
        th_eigenenergies.append(e)
        th_eigenenergies.append(-e)
    
    th_eigenenergies = sorted(th_eigenenergies)
    
    
    print(tabulate(table, headers=["Nº", "Solution", "Distance to 0"]))
    print("Número de soluciones de bulk = ", len(bulk))
    print("Número de soluciones encontradas =", len(roots)) 
    return roots, th_eigenenergies, bulk




def th_eigenenergies_check(t1,t2,d,N):
    start_time = time.time() # Comienzo de la medición de tiempo 
    th_eigenenergies = find_roots(N//2,d)[1]
    eigenenergies = dospar(t1,t2,N)[0]/t1

#    print("th_eigenenergies",th_eigenenergies)
#    print("eigenenergies",sorted(eigenenergies))
    print(len(eigenenergies))
    print(len(th_eigenenergies))
    
 
    df = pd.DataFrame({"Theoretical eigenenergies": th_eigenenergies, "Numerical eigenenergies": sorted(eigenenergies)})
    df["$\Delta E=|E_{th}-E_{num}|$"] = abs(df["Theoretical eigenenergies"]-df["Numerical eigenenergies"])
    
    df.to_csv('tabla_comparativa.csv')
    elapsed_time = time.time() - start_time # Tiempo de ejecución
    print(f"Tiempo de ejecución de la función {th_eigenenergies_check.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución

  

#######################################################################
#######################################################################
"""INTERACCIÓN ELECTRÓN ELECTRÓN"""

def Vbulk(U,M,D):
    sol = find_roots(N//2,d)[2]
   
    def A_alpha(k):
        return 1/np.sqrt(M+0.5*(1-np.sin((2*M+1)*k)/np.sin(k)))
    def suma(k1,k2):
        suma = 0
        for i in range(1,M+1):
            suma += (np.sin(k1*i))**(2)*(np.sin(k2*i))**(2)
        return suma
    
    def suma2(k1,k2):
#        res = (4*M-2*(np.sin((2*M+1)*k1)/(np.sin(k1))+np.sin((2*M+1)*k2)/(np.sin(k2)))+(np.sin((2*M+1)*(k1+k2))/(np.sin((k1+k2)))+np.sin((2*M+1)*(k1-k2))/np.sin(k1-k2))+2)
        res2 = 2*(2*M+1-(np.sin((2*M+1)*k1)/np.sin(k1)))+2*(2*M+1-(np.sin((2*M+1)*k2)/np.sin(k2)))-(2*M+1-(np.sin((2*M+1)*(k1-k2))/np.sin(k1-k2)))-(2*M+1-(np.sin((2*M+1)*(k1+k2))/np.sin(k1+k2)))
        return res2/16
    
    
    combinaciones = list(itertools.combinations(sol, 2))
    comblist = [list(elem) for elem in combinaciones]
    
    
    mejor_comb = comblist[0]
    mejor_dif = abs(sum(mejor_comb)-np.pi)
    for comb in combinaciones[1:]:
        sumal = sum(comb)
        dif = abs(sumal-np.pi)
        if dif<mejor_dif:
            mejor_comb = comb
            mejor_dif = dif
    
    #Seleccionamos dos k's al azar (los dos autoestados que interactúan)
#    k1 = random.choice(sol)
    k1 = mejor_comb[0]
    print("k1=",k1)
#    k2 = random.choice(sol)
    k2 = mejor_comb[1]
    print("k2=",k2) 
    
            
    print ("La combinación de k's más cercana a pi es ", mejor_comb)
    print("Diferencia a pi", mejor_dif)
    print("Numero combinaciones posibles", len(comblist))
    print("Suma simplificada=", suma2(k1,k2))
#    print("Suma simplificada Jaime=", suma2(k1,k2)[1])
    print("Suma=",suma(k1,k2))
    
    
    

def Vedge(U,M,D):
    qlist = find_roots(N//2,d)[0]
    q = qlist[-1]
    print(q)
    def suma(q): 
        suma = 0 
        for i in range(1,M+1):
            suma += (np.sinh(i*q))**4
        return suma
    
    
    print("Suma = ", suma(q))
    print("Suma simplificada = ", suma2)      




"""Código fuera de las funciones"""  
#Vedge(1,N//2,d)    

q = 0.12
e=math.e
M = N//2
suma = 0
suma2 = 0 
for i in range(1,M+1):
    suma += 8*(np.sinh(i*q))**4
print("Suma = ", suma)


suma2 = 3*M+0.5*(np.sinh(q*2*(2*M+1))/np.sinh(2*q)-1) -2*(np.sinh(q*(2*M+1))/np.sinh(q)-1)
print("Suma simplificada = ", suma2)  


#Vbulk(1,N//2,d)
#th_eigenenergies_check(t1,t2,d,N)
#plot_energias(t1,t2,N)    
#find_roots(N//2,d)
#print(tridiagfinal(t1,t2,N))
#plot_eigenstates(t1,t2,N)
#plt.show()    
        
    

    

#def find_roots(M, D, a=0+10**(-6), b=np.pi-10**(-6), eps=1e-7, max_iter=10000, n=100000):
#    """
#    Encuentra todas las raíces de la función f en el intervalo [a, b] utilizando el método de Newton.

#    Args:
#    - M: un parámetro para la función f.
#    - D: un parámetro para la función f.
#    - a: el límite inferior del intervalo de búsqueda.
#    - b: el límite superior del intervalo de búsqueda.
#    - eps: la tolerancia para la convergencia del método de Newton.
#    - max_iter: el número máximo de iteraciones permitidas en el método de Newton.
#    - n: el número de subintervalos en los que se divide el intervalo [a, b] para buscar las raíces.

#    Returns:
#    - Una lista que contiene todas las raíces encontradas en el intervalo [a, b].
#    """
#    
#    
#    #Ecuación para edge states    
#    def fq(x):
#        return np.tanh(x*(M+1))-np.sinh(x)/(np.cosh(x)-1/D)
#        
#    #Ecuación para bulk states    
#    def f(x):
#        return np.tan(x*(M+1))-D*np.sin(x)/(1+D*np.cos(x))
#        
#    def df(x):
#        return ((M + 1) * (1 / np.cos((M + 1) * x))**2) - ((D**2 * np.sin(x)**2) / (D * np.cos(x) + 1)**2) - ((D * np.cos(x)) / (D * np.cos(x) + 1))



#    roots = []
#        
#    dx = (b - a)/n
#    
#    for i in range(n):
#        x0 = a + i*dx
#        x1 = a + (i+1)*dx
#        x = x0
#        
#        for j in range(max_iter):
#            fx = f(x)
#            
#            if abs(fx) < eps:
#                roots.append(x)
#                break
#                
#            dfx = df(x)
#            
#            if dfx == 0:
#                break
#                
#            x = x - fx/dfx
#            
#            if x < x0 or x > x1:
#                break
#    

#    qsol = bisect(fq,a,b,xtol=1e-15)
#    roots.append(qsol) #El último elemento de la lista roots es la solución de edge
#    print("QSOL",qsol)
#    print("len",len(roots))
#    
#        

#    table = []
#    th_eigenenergies = []
#    
#    for i in roots:
#        if i == roots[-1] and len(roots)==M:
#            dif=fq(i)
#            arg = 1+D**2-2*abs(D)*np.cosh(i)
#            print("arg",arg)
#            e = np.sqrt(arg)
#           
#        else: 
#            dif=f(i)
#            e = np.sqrt(1+D**2+2*D*np.cos(i))
#            
#        sublist=[roots.index(i)+1,i,dif]
#        table.append(sublist)
#        
#        th_eigenenergies.append(e)
#        th_eigenenergies.append(-e)
#    
#    th_eigenenergies = sorted(th_eigenenergies)
#    
#    

#    print(tabulate(table, headers=["Nº", "Solution", "Distance to 0"]))
#    print("Número de soluciones encontradas =", len(roots))    
##    print(th_eigenenergies)

#    
#            
#    return roots, th_eigenenergies


    
    
#def plot_eigenstates(t1,t2,N):
#    start_time = time.time() # Comienzo de la medición de tiempo 
#    X_par=[i for i in range(2, N+1,2)]
#    X_impar=[i for i in range(1,N+1,2)]
#    if N%2==0:
#        _, eigenvectors, _, _ = dospar(t1,t2,N)
#    else: 
#        _, eigenvectors = impar(t1,t2,N)
#        
#    for j in range(0,N):
##        print(eigenvectors[j])
#        pesos_b=[]
#        pesos_a=[]
#        for i in range(0,N):
#            if i%2==0:
#                pesos_a.append(eigenvectors[i][j])
#            else:
#                pesos_b.append(eigenvectors[i][j])
#            i+=1
#        plt.figure()
#        print("Pesos átomos A",pesos_a)
#        print("Pesos átomos B",pesos_b)
#        plt.bar(X_par,pesos_b,color="red", edgecolor="black",label="B atoms")
#        plt.bar(X_impar,pesos_a,color="blue", edgecolor="black", label="A atoms")
#        plt.title("Eigenstate number " + str(j+1) + " for $t_1$=" + str(t1) + ", $t_2$=" + str(t2))

#        plt.legend()
#        plt.xlabel("Atom index")
#        plt.ylabel("Eigenstate weight")
#        if (not os.path.exists(ruta_N)):
#            os.mkdir(ruta_N)
#        print(j)
#        outputplot=ruta_N+"/"+"Eigenstate"+str(j+1)+".png"
#        plt.savefig(outputplot)
#  
#    j+=1

    
#    for j in range(0,2*N):
##        print(eigenvectors[j])
#        pesos_b=[]
#        pesos_a=[]
#        for i in range(0,2*N):
#            if i%2==0:
#                pesos_a.append(eigenvectors[i][j])
#            else:
#                pesos_b.append(eigenvectors[i][j])
#            i+=1
#        plt.figure()
#        print("Pesos átomos A",pesos_a)
#        print("Pesos átomos B",pesos_b)
#        plt.bar(X_par,pesos_b,color="red", edgecolor="black",label="B atoms")
#        plt.bar(X_impar,pesos_a,color="blue", edgecolor="black", label="A atoms")
#        plt.title("Eigenstate number " + str(j+1) + " for $t_1$=" + str(t1) + ", $t_2$=" + str(t2))

#        plt.legend()
#        plt.xlabel("Atom index")
#        plt.ylabel("Eigenstate weight")
#        if (not os.path.exists(ruta_N)):
#            os.mkdir(ruta_N)
#        print(j)
#        outputplot=ruta_N+"/"+"Eigenstate"+str(j+1)+".png"
#        plt.savefig(outputplot)
#   
#    j+=1    
    
    



    
    
    









#######################################################################
#######################################################################

##(DES)COMENTAR PARA (NO) REPRESENTAR TODOS LOS EIGENSTATES DE UNA VEZ 
#for j in range(0,2*N):
#    print(eigenvectorsa[j])
#    pesos_b=[]
#    pesos_a=[]
#    for i in range(0,2*N):
#        if i%2==0:
#            pesos_a.append(eigenvectorsa[i][j])
#        else:
#            pesos_b.append(eigenvectorsa[i][j])
#        i+=1
#    plt.figure()
##    print("Pesos átomos A",pesos_a)
##    print("Pesos átomos B",pesos_b)
#    plt.bar(X_par,pesos_b,color="red", edgecolor="black",label="B atoms")
#    plt.bar(X_impar,pesos_a,color="blue", edgecolor="black", label="A atoms")
#    plt.title("Eigenstate number " + str(j) + " for $t_1$=" + str(t1) + ", $t_2$=" + str(t2))

#    plt.legend()
#    plt.xlabel("Atom index")
#    plt.ylabel("Eigenstate weight")
#    if (not os.path.exists(ruta_N)):
#        os.mkdir(ruta_N)
#    outputplot=ruta_N+"/"+"Eigenstate"+str(j)+".png"
#    plt.savefig(outputplot)
#   
#    j+=1
#    
#(DES)COMENTAR PARA (NO) REPRESENTAR UN ÚNICO EIGENSTATE j   
#pesos_b=[]
#pesos_a=[]
#j=4
#for i in range(0,2*N):
#    if i%2==0:
#        pesos_b.append(eigenvectorsa[j][i])
#    else:
#        pesos_a.append(eigenvectorsa[j][i])
#    i+=1
#print("Pesos átomos A",pesos_a)
#print("Pesos átomos B",pesos_b)
#plt.figure()
#plt.bar(X_par,pesos_b,color="red", label="B atoms")
#plt.bar(X_impar,pesos_a,color="blue", label="A atoms")
#plt.title("Eigenstate number "+str(j))
#plt.legend()
#plt.xlabel("Atom index")
#plt.ylabel("Eigenstate weight")
#######################################################################
#######################################################################
#plot_eigenstates(t1,t2,N)
#plt.show()
#    def f(x):
#        return np.tan(x*(M+1))-D*np.sin(x)/(1+D*np.cos(x))
#    def df(x):
#        return (M+1)*(1/np.cos((M+1)*x))**2 - D**2*(np.sin(x))**2/(D*np.cos(x)+1)**2 - D*np.cos(x)/(D*np.cos(x)+1)


