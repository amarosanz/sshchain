######################################################################
######################################################################
#Programa para calcular las autoenergías y los autoestados de una cadena SSH finita de N celdas unidad 
import sys, os
import numpy as np
import numpy.linalg as linalg
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import math
import time



######################################################################
######################################################################

#Definición constantes
t1=0.7 #integral de salto intra-cell   
t2=0.3 #integral de salto inter-cell
d=t2/t1
N =21 #número de átomos de la cadena 

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

        
    
 
    
    
    
    
  
plot_eigenstates(t1,t2,N)
#plt.show()    
        
    

    
    
    
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



