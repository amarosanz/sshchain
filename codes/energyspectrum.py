######################################################################
######################################################################
#Programa para representar el espectro de las autoenergías a t2 fijado y t1 variable 

import sys, os
import numpy as np
import numpy.linalg as linalg
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

#Definición constantes
t1=1 #integral de salto intra-cell   
#t2=1 #integral de salto inter-cell
N =10 #número de celdas unidad de la cadena

rutassh = os.path.join(os.getcwd(), "plots_ssh")
ruta_plots= rutassh+"/"+"energyspectrum"
######################################################################
######################################################################

#Construcción del hamiltoniano tridiagonal por bloques
def tridiag(c, u, d, N): 
    # c, u, d are center, upper and lower blocks, repeat N times
    cc = block_diag(*([c]*N))
    shift = c.shape[1]
    uu = block_diag(*([u]*N)) 
    uu = np.hstack((np.zeros((uu.shape[0], shift)), uu[:,:-shift]))
    dd = block_diag(*([d]*N)) 
    dd = np.hstack((dd[:,shift:],np.zeros((uu.shape[0], shift))))
    return cc+uu+dd
    
def tridiagfinal(t1,t2,N):
    H0 = np.array([[0, t1], [t1, 0]])
    H1 = np.array([[0, 0], [t2, 0]])
    HM1 = np.array([[0, t2], [0, 0]])
    H = tridiag(H0,H1,HM1,N)
    return H


######################################################################
######################################################################

M=3 #hasta qué valor de t2 quieres plotear
precision= 100 #cuantos valores de t2 quieres plotear
t2list=np.linspace(0,M,precision)
print("t2list:", t2list)

lista_energias=[]

#HOLD t1 constant
#for i in t2list: 
#    H = tridiagfinal(t1,i,N)
#    eigenvalues, eigenvectors = np.linalg.eigh(H)
#    energia=eigenvalues.tolist()
#    lista_energias.append(energia)
#    

#HOLD t2 constant 
for i in t2list: 
    H = tridiagfinal(t1,i,N)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    energia=eigenvalues.tolist()
    lista_energias.append(energia)

energias_ordenadas = [[] for i in range(len(lista_energias[0]))]
for i in lista_energias:
    for j, elemento in enumerate(i):
        energias_ordenadas[j].append(elemento)

#print("Lista energias sin ordenar", lista_energias)
#print("Energias ordenadas:",energias_ordenadas)

for i in range(0,2*N):
#    pos=lista_energias.index(i)
    plt.plot(t2list,energias_ordenadas[i],color="blue",lw=1)

plt.xlabel("$t_2$")
plt.ylabel("Eigenenergies")
plt.title("Eigenvalue spectrum vs $t_2$, $t_1$="+str(t1))
plt.grid(visible=True,lw=0.5)

if (not os.path.exists(ruta_plots)):
        os.mkdir(ruta_plots)   

outputplot=ruta_plots+"/"+"t1-"+str(t1)+"N-"+str(N)+".pdf"
plt.savefig(outputplot)        
        
         
#plt.show()
        
#        pos=lista_energias.index(j)
#        print(pos)
#        energias_ordenadas.append(lista_energias[pos][i])
#        print(energias_ordenadas) 
        
   
#plt.show()
    
#PLOT EIGENENERGIES
#plt.figure()
#x = np.linspace(1, 2*N+1, 2*N)
#plt.scatter(x,sorted(eigenvaluesb), label="$t_1$="+str(t2)+", $t_2$="+str(t1),color="red", s=5)
#plt.scatter(x,sorted(eigenvaluesa), label="$t_1$="+str(t1)+", $t_2$="+str(t2), s=5)
#plt.ylabel("Eigenenergies")
#plt.xlabel("Atoms")
#plt.legend()

    
##Cálculo de las eigenenergies y los eigenstates
#eigenvaluesa, eigenvectorsa = np.linalg.eigh(Ha)
#eigenvaluesb, eigenvectorsb = np.linalg.eigh(Hb)
