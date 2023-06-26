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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import pandas as pd
from tabulate import tabulate
import math
import time
import random
import itertools


######################################################################
######################################################################
#Definición constantes 

t1 = 1.2 #integral de salto intra-cell   
t2 = 1 #integral de salto inter-cell
D=t2/t1
N = 20 #número de átomos de la cadena 
M = N//2 #número de celdas unidad de la cadena
Mc = D/(1-D) #número de celdas unidad crítico de la cadena
dop = 2 #dopaje estados de borde
U = 1.5 #correlation
kb = 8.617333262e-5 #eV/K (Constante de Boltzmann) 

par = N % 2 == 0
topo = False

print("\nCADENA {}DE N = {} ÁTOMOS".format("PAR " if par else "IMPAR ", N))
print("\nMc =", Mc)
print("t1 =", t1)
print("t2 =", t2)
print("D = t2/t1 =", D)

if D > 1:
    print("\nFASE TRIVIAL: 0 ESTADOS DE BORDE ESPERADOS")
else:
    if D < 1 and M > Mc:
        if par:
            print("\nFASE TOPOLÓGICA: 2 ESTADOS DE BORDE ESPERADOS")
        else:
            print("\nFASE TOPOLÓGICA: 1 ESTADO DE BORDE ESPERADO")
        topo = True
    else:
        print("\nFASE TRIVIAL: 0 ESTADOS DE BORDE ESPERADOS")





parent_dir = os.path.dirname(os.getcwd())
if (not os.path.exists(os.path.join(parent_dir, "plots_ssh"))):
    os.mkdir(os.path.join(parent_dir, "plots_ssh"))
ruta = os.path.join(parent_dir, "plots_ssh")
ruta_N= ruta+"/"+str(N)+"chain"

######################################################################
######################################################################
#Construcción del Hamiltoniano tridiagonal por bloques 


    
def hfinal(t1,t2,N):
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
    print(f"\nTiempo de ejecución de la función {hfinal.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución
    
    return matriz_final 
    
    
######################################################################
######################################################################

#CÁLCULO EIGENVALUES Y EIGENSTATES

def eigen(t1,t2,N):
    start_time = time.time() # Comienzo de la medición de tiempo 
    h=hfinal(t1,t2,N)
    values, states = np.linalg.eigh(h)
    idx = values.argsort()[::-1]   
    values = values[idx]
    states = np.transpose(states[:,idx])
#    print("Energies: ", values)
#    print("States: ", states)
#    suma_fila = np.sum(np.abs(states)**2, axis=1)  #Comprueba la normalización de los autoestados
#    print(suma_fila)
    elapsed_time = time.time() - start_time # Tiempo de ejecución
    print(f"Tiempo de ejecución de la función {eigen.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución
    
    return values,states
    
def ks(t1, t2, N):
    D = t2 / t1
    par = N % 2 == 0
    topo = False
    k = []
    
    if par:
        if D > 1:
            topo = False
        elif D < 1: 
            if M < Mc:
                topo = False
            elif M > Mc: 
                topo = True 
    else:
        topo = True
        par = False 

    h = hfinal(t1, t2, N)
    values, states = np.linalg.eigh(h)

    for v in values.tolist():
        pos = values.tolist().index(v)
        x = ((v / t1) ** 2 - 1 - D ** 2) / (2 * D)

        if par and topo:
            if pos != M - 1 and pos != M:
                kv = np.arccos(x)
            else:
                kv = np.pi
        elif par and not topo: 
            kv = np.arccos(x)
        else: 
            if pos != M:
                kv = np.arccos(x)
            else: 
                kv = np.pi

        k.append(kv)

    return k

#######################################################################
#######################################################################

#PLOTS 

#EIGENENERGIES
def plot_energias(t1,t2,N):
    start_time = time.time() # Comienzo de la medición de tiempo
    h=hfinal(t1,t2,N)
    values, states = np.linalg.eigh(h)
    #print("states: ", states)
    k = ks(t1,t2,N)
    print("k:    ", k)        
    plt.figure()
    plt.ylabel("E")
    plt.xlabel("k")
    kspace = np.linspace(0,np.pi,1000)
    E = t1*np.sqrt(1+D**2+2*D*np.cos(kspace))    
    plt.plot(kspace, E)
    plt.plot(kspace, -E)
    
    plt.scatter(k,values, s=17, color="black")
    plt.title(str(N)+"-atom chain eigenvalues")
    plt.grid(visible=True,lw=0.5)
#    plt.legend(fontsize=13)
    plt.show()
#    if (not os.path.exists(ruta_N)):
#        os.mkdir(ruta_N)
#    outputp=ruta_N+"/"+"eigenvaluesk.png"
#    plt.savefig(outputp)
        
    elapsed_time = time.time() - start_time # Tiempo de ejecución
    print(f"Tiempo de ejecución de la función {plot_energias.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución     


def eigenstates(t1,t2,N):
    start_time = time.time() # Comienzo de la medición de tiempo 
    energies, states = eigen(t1,t2,N)
    x=[i for i in range(1, N+1)]
 
    #    print("eigenvectors=",eigenvectors)
    fig, ax = plt.subplots()
    scatter = ax.scatter(np.tile(np.arange(states.shape[0]), states.shape[1]), np.repeat(energies, states.shape[0]), c=states.flatten(),cmap='jet')
    
    # Establecer límites personalizados del colormap
    cmin = -0.75  # Límite mínimo
    cmax = 0.75  # Límite máximo
    scatter.set_clim(cmin, cmax)
    # Configurar colorbar vertical a la derecha del gráfico
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', ticks=[-0.5,0,0.5], pad=0.05)
    cbar.set_label("$\Psi_n$",rotation=0, labelpad=20, size = 17)  # Rotar y ajustar el espaciado de la etiqueta)


    # Invertir etiqueta de la colorbar y ajustar tamaño
    cbar.ax.invert_yaxis()
    cbar.ax.tick_params(labelsize=14)  # Ajustar el tamaño de las etiquetas de la colorbar

    # Configurar ejes
    ax.set_xticks(np.arange(states.shape[0]))
    ax.set_xticklabels(np.arange(states.shape[0]) + 1)
    ax.set_ylabel('E')
    ax.set_xlabel('n')
    plt.xticks([0, N-1])
    plt.subplots_adjust(right=0.85)  # Ajustar la posición de la colorbar a la derecha

    plt.show()
    

    
    elapsed_time = time.time() - start_time # Tiempo de ejecución
    print(f"Tiempo de ejecución de la función {eigenstates.__name__}: {elapsed_time:.5f} segundos") # Imprime el tiempo de ejecución


    
def plot_all(t1, t2, N):
    h = hfinal(t1, t2, N)
    k = ks(t1, t2, N)
    values, states = np.linalg.eigh(h)

    # Crear la figura y los subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, figsize=(12, 12))

    # Ajustar los márgenes y el espacio entre subplots
    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    # Configurar tamaño de fuente de los ejes
    font_size = 14
    ax1.set_ylabel("E", fontsize=font_size)
    ax3.set_xlabel("k", fontsize=font_size)
    ax3.set_ylabel("E", fontsize=font_size)
    ax4.set_xlabel("n", fontsize=font_size)

    kspace = np.linspace(0, np.pi, 1000)
    E = t1 * np.sqrt(1 + D ** 2 + 2 * D * np.cos(kspace))
    ax1.plot(kspace, E, color="blue", label = "N=\u221e")
    ax1.plot(kspace, -E, color="blue")

    ax1.scatter(k, values, s=17, color="black",  label="N=" + str(N) + ", $t_1$=" + str(round(t1,2)) + ", $t_2$=" + str(round(t2,2)))
    ax1.set_title("Dispersion relation", size = 18)
    ax1.set_yticks([-2, -1,0,1,2])
    ax1.grid(visible=True, lw=0.5)

    energies, states = eigen(t1, t2, N)
    x = np.arange(1, N+1)
    scatter = ax2.scatter(np.tile(x, states.shape[1]), np.repeat(energies, states.shape[0]), c=states.flatten(), cmap='jet')

    cmin = -0.75
    cmax = +0.75
    scatter.set_clim(cmin, cmax)

    cbar = fig.colorbar(scatter, ax=ax2, orientation='vertical', ticks=[-0.5, 0, 0.5], pad=0.05)
    cbar.set_label("$\Psi_n$", rotation=0, labelpad=20, size=17)
#    cbar.ax.invert_yaxis()
    cbar.ax.tick_params(labelsize=14)


    # Configurar ejes
    ax2.set_xticks(np.arange(states.shape[0]))
    ax2.set_xticks([1, N])


    # Añadir título al plot de la derecha
    ax2.set_title("Eigenstates", size = 18)

    # Subplots con t1 y t2 invertidos
    h_inv = hfinal(t2, t1, N)
    k_inv = ks(t2, t1, N)
    values_inv, states_inv = np.linalg.eigh(h_inv)

    ax3.plot(kspace, E, color="blue")
    ax3.plot(kspace, -E ,color="blue", label = "N=\u221e")
    ax3.scatter(k_inv, values_inv, s=17, color="black", label="N=" + str(N) + ", $t_1$=" + str(round(t2,2)) + ", $t_2$=" + str(round(t1,2)))

    ax3.grid(visible=True, lw=0.5)
    ax3.set_yticks([-2,-1,0,1,2])

    energies_inv, states_inv = eigen(t2, t1, N)
    scatter_inv = ax4.scatter(np.tile(x, states_inv.shape[1]), np.repeat(energies_inv, states_inv.shape[0]), c=states_inv.flatten(), cmap='jet')

    scatter_inv.set_clim(cmin, cmax)

    cbar_inv = fig.colorbar(scatter_inv, ax=ax4, orientation='vertical', ticks=[-0.5, 0, 0.5], pad=0.05)
    cbar_inv.set_label("$\Psi_n$", rotation=0, labelpad=20, size=17)
    cbar_inv.ax.tick_params(labelsize=14)


    # Configurar ejes
    ax4.set_xticks(np.arange(states.shape[0]))
    ax4.set_xlabel('n')
    ax4.set_xticks([1, N])
    ax4.set_xlabel('n')
    
    # Colocar leyenda en el medio izquierdo
    ax1.legend(loc="center left", fontsize = 13)
    ax3.legend(loc="center left", fontsize = 13)
    if (not os.path.exists(ruta_N)):
        os.mkdir(ruta_N)
    outputplot=ruta_N+"/"+"Allopencell.pdf"
    plt.savefig(outputplot)    
#    plt.show()
        
#######################################################################
#######################################################################      
#ENCONTRAR SOLUCIONES DE VOLUMEN Y DE BORDE 
def find_roots(M, D):
    qlim = np.log(1/D)
    a_prev = np.pi/(2 * M)
    roots = []
    bulk = []
    table = []
    p = []
    # Ecuación para edge states
    def fq(x):
        return np.tanh(x * M) - D * np.sinh(x) / (1 - D * np.cosh(x))

    # Ecuación para bulk states
    def f(x):
        return np.tan(x * M) + D * np.sin(x) / (1 + D * np.cos(x))

    for i in range(2, M + 1):
        a = np.pi * (i - 0.5) / M
        p.append(a)

        while f(a_prev) > 0:
            a_prev += 1e-8
        if M < Mc and D < 1 and i == M:
            a = np.pi
        while f(a) < 0:
            a -= 1e-8           
        if a > np.pi:
            break
        
        sol = bisect(f, a_prev, a, xtol=1e-14)
        if abs(f(sol)) < 1e-2:
            roots.append(sol)
            if i != M + 1:
                bulk.append(sol)
            dif = f(sol) if i != M + 1 else fq(sol)
            sublist = [roots.index(sol) + 1, sol, dif]
            table.append(sublist)
        a_prev = a
        

    if D < 1 and M > Mc:
        qsol = bisect(fq, 0 + 10 ** (-8), qlim, xtol=1e-20)
        if abs(fq(qsol)) < 1e-4:
            roots.append(qsol)
            qenergy = np.sqrt(1+D**2-2*D*np.cosh(qsol))
        bulk = roots[:-1]
    
    if len(roots) == M-1:  #Para calcular la solución que falta en la fase trivial
        table = []
        l = np.arccos(-1/D)  
        p.append(l)
        p.sort() 
        smax = -1000 
        tol = 1e-7
        for i in range(len(roots)-1):
            s = abs(roots[i]-roots[i+1])
            if s > smax:
                smax = s       
                a = roots[i]
                a = p[i]+tol

                b = roots[i+1]
                b = p[i+1]-tol
                fi = i+2


        solm = bisect(f, a, b, xtol = 1e-14)  

        roots.append(solm)
        bulk.append(solm)
        roots.sort()
        bulk.sort()
        for el in roots: 
            sub = [roots.index(el)+1,el,f(el)]
            table.append(sub)
            

    bulkenergies = [np.sqrt(1 + D ** 2 + 2 * D * np.cos(sol)) for sol in bulk]

    allenergies = bulkenergies.copy()  # Copiar los elementos de bulkenergies a allenergies

    if topo:
        allenergies.append(qenergy)

    # Agregar los negativos de bulkenergies a allenergies
    allenergies.extend([-e for e in allenergies])
    bulkenergies.extend([-e for e in bulkenergies])  # Agregar los negativos de bulkenergies a bulkenergies

    allenergies.sort()
    bulkenergies.sort()

    headers = ["Nº", "Solution", "Distance to 0"]
    table.sort(key=lambda x: x[1])  # Ordenar la tabla por la solución de menor a mayor
    print("\nNúmero de soluciones de bulk =", len(bulk))
    print("Número de soluciones encontradas =", len(roots))
    print("\nSoluciones de bulk:")
    print(tabulate(table, headers=headers))
    if topo == True:
        print("Solución de borde encontrada en k = PI -"+str(roots[-1])+"j")

    return roots, allenergies, bulk, bulkenergies





def correlation(U,t1,t2,N,dop):           
    def Ak(k):
        Aksquared = 1/(M+0.5*(1-np.sin((2*M+1)*k)/np.sin(k)))
        return Aksquared
        
    rootslist, _, klist, _ = find_roots(M, D)
    klist = np.array(klist)  # Convertir klist en un array NumPy
    print("\nrootslist, ", rootslist)
    print("\nklist", klist)


    if topo:
        q = rootslist[-1]
        Aq = (0.5 * (np.sinh(q * (2 * M + 1)) / np.sinh(q) - 1) - M) ** (-1 / 2)
        sin_k = np.sin(klist[:, np.newaxis] * np.arange(1, M + 1))
        sinh_q = np.sinh(q * np.arange(1, M + 1))
        ube = 4 * U * Ak(klist) * Aq ** 2 * np.sum(sin_k ** 2 * sinh_q ** 2, axis=1)
        ue = 4 * U * Aq ** 4 * np.sum(sinh_q ** 4) 
        print("Ube", ube)
        print("Ue", ue)


    ub = np.zeros((len(klist), len(klist)))
    for i, k1 in enumerate(klist):
        Ak1squared = Ak(k1)
        for j, k2 in enumerate(klist):
            Ak2squared = Ak(k2)
            element = 0.0
            for n in range(1, M+1):
                element += np.sin(n * k1) ** 2 * np.sin(n * k2) ** 2
            ub[i, j] = 4*Ak1squared*Ak2squared*U*element
            
    for i, row in enumerate(ub):
        row_sum = np.sum(row)
        print("La suma en la fila", i+1, "=", row_sum)
        
    print("ub",ub)
    
    return ub, ube, ue


    
def plotpm(U, t1, t2, N, dop):
    ub, ube, ue = correlation(U, t1, t2, N, dop)
    _, ewoc, _, ewocb = find_roots(M, D)
    print("\nEWOC =", ewoc)
    
    ewc = np.zeros(2 * M)  # Inicializar array "ewc" con ceros

    for i in range(2 * M):
        if i == M - 1 or i == M: 
            ewc[i] = ewoc[i] + ue * (1 + dop / 2) + np.sum(ube)
        else:
            row_index = i % (M - 1) 
            ewc[i] = ewoc[i] + np.sum(ub[row_index]) + ube[row_index] * (1 + dop / 2)

    print("EWC:", ewc)
    print("\nEWC-EWOC:", ewc - ewoc)
    
    ewoc = np.repeat(ewoc, 2)
    ewc = np.repeat(ewc, 2)
    
    markers = ['v', '^'] * (len(ewoc) // 2)  # Lista alternada de marcadores
    colors = ['blue', 'red'] * (len(ewoc) // 2)  # Lista alternada de colores
    
    din = 0.7
    dout = (4 * M -1 - 2 * M * din) / (2 * M - 1)
    d = [1]
    pos = d[0]
    for i in range (2 * N -1):
        if i % 2 == 0:
            pos += din
            d.append(pos)
        else:
            pos += dout
            d.append(pos)
            
    for i in range(len(ewoc)):
        plt.scatter(d[i], ewoc[i], marker=markers[i], color=colors[0], label="Eigenenergies w/o correlation" if i == 0 else None)
        plt.scatter(d[i], ewc[i], marker=markers[i], color=colors[1], label="Eigenenergies w correlation (PM)" if i == 0 else None)
    


    
    plt.title(str(N) + "-atom chain" + ", $t_1$=" + str(round(t2, 2)) + ", $t_2$=" + str(round(t1, 2)) + ", U/$t_1$=" + str(U) + ", $\delta$=" + str(dop), fontsize = 16)
    plt.ylabel("E/$t_1$", fontsize = 15)
    plt.xlabel("State index", fontsize = 15)
    plt.grid(visible=True, lw=0.5)
    plt.legend(fontsize = 15)
    
    inset_ax = inset_axes(plt.gca(), width="30%", height="30%", loc='lower right')
    inset_ax.set_title("Edge states zoom", fontsize=12)
        # Customize the position and size of the inset plot as per your requirement
    inset_ax.set_xlim(18, 23)  # Set the x-axis limits for the inset plot
    inset_ax.set_ylim(-0.5, 2)  # Set the y-axis limits for the inset plot
    inset_ax.set_xticklabels([])  # Set the x-axis tick positions for the inset plot
    inset_ax.set_yticks([-0.5, 0.7, 2])  # Set the y-axis tick positions for the inset plot
# Plot the data for the inset plot
    for i in range(len(ewoc)):
        inset_ax.scatter(d[i], ewoc[i], marker= markers[i], color='blue', label="Eigenenergies w/o correlation")
        inset_ax.scatter(d[i], ewc[i], marker= markers[i], color='red', label="Eigenenergies w correlation (PM)")
    
    if (not os.path.exists(ruta_N)):
        os.mkdir(ruta_N)
    outputp=ruta_N+"/"+"pmdop1.png"
    plt.savefig(outputp)


"""SOLUCIÓN FERROMAGNÉTICA"""
        

#def plotfm(U, t1, t2, N, dop):
#    ub, ube, ue = correlation(U, t1, t2, N, dop)
#    _, ewoc, _, ewocb = find_roots(M, D)
#    print("\nEWOC =", ewoc)
#    
#    ub = np.repeat(ub,2, axis =0)
#    ube = np.repeat(ube,2)
#    print("ub", ub)
#    print("ube", ube)
#    ewc = np.zeros(2 * M)  # Inicializar array "ewc" con ceros
#    ewoc = np.repeat(ewoc, 2)
#    ewc = np.repeat(ewc, 2)
#    
#    for i in range(4 * M):
#        print("\ni = ", i)
#        if i%2 == 0: 
#            if i == 2*(M - 1) or i == 2*M: 
#                ewc[i] = ewoc[i]  + 2 * ue + np.sum(ube)/2
#            else:
#                row_index = i % (2 * (M - 1))
#                print("row_index = ", row_index) 
#                print("ube[row_index] =", ube[row_index])
#                ewc[i] = ewoc[i] + np.sum(ub[row_index]) + 2*ube[row_index]
#        else: 
#            if i == 2*(M - 1) or i == 2*M: 
#                ewc[i] = ewoc[i] + np.sum(ube)/2
#            else:
#                row_index = i % (2 * (M - 1)) 
#                ewc[i] = ewoc[i] + np.sum(ub[row_index]) 
#            

#    print("EWC:", ewc)
#    print("\nEWC-EWOC:", ewc - ewoc)
#    

#    
#    markers = ['v', '^'] * (len(ewoc) // 2)  # Lista alternada de marcadores
#    colors = ['blue', 'red'] * (len(ewoc) // 2)  # Lista alternada de colores
#    
#    din = 0.5
#    dout = (2 * M - M * din) / M
#    
#    d = 0 
#    for i in range(len(ewoc)):
#        if i % 2 == 0:
#            d += dout
#        else: 
#            d += din
#        plt.scatter(d, ewoc[i], marker=markers[i], color=colors[0], label="Eigenergies w/o correlation" if i == 0 else None)
#        plt.scatter(d, ewc[i], marker=markers[i], color=colors[1], label="Eigenenergies w correlation" if i == 0 else None)
##        print("d = ", d)
#    plt.title(str(N) + "-atom chain eigenvalues" + ", $t_1$=" + str(round(t1, 2)) + ", $t_2$=" + str(round(t2, 2)) + ", U/$t_1$=" + str(U) + ", $\delta$=" + str(dop))
#    plt.grid(visible=True, lw=0.5)
#    plt.legend()
#    plt.show()



def plotfm(U, t1, t2, N, dop):
    ub, ube, ue = correlation(U, t1, t2, N, dop)
    _, ewoc, _, ewocb = find_roots(M, D)

    print("\nEWOC =", ewoc)
    
    dwc = np.zeros(2 * M)  # Inicializar array "ewc" con ceros
    uwc = np.zeros(2 * M)
     
    for i in range(2 * M):
        if i == M - 1 or i == M:
            dwc[i] = ewoc[i]  + 2 * ue + np.sum(ube)
        else:
            row_index = i % (M - 1) 
            dwc[i] = ewoc[i] + np.sum(ub[row_index]) + 2*ube[row_index]
        print("ewoc[i] = ", ewoc[i]) 
        
    for i in range(2 * M):
        if i == M - 1 or i == M: 
            uwc[i] = ewoc[i] + np.sum(ube) + ue * dop 
        else:
            row_index = i % (M - 1) 
            uwc[i] = ewoc[i] + np.sum(ub[row_index]) + dop * ube[row_index]
        print("ewoc[i] = ", ewoc[i])             
               
    
    print("DWC:", dwc)
    print("UWC:", uwc)
    ewoc = np.repeat(ewoc, 2)
    ewc = [item for pair in zip(dwc, uwc) for item in pair]
    print("EWC: ", ewc)
    print("\nEWC-EWOC:", ewc - ewoc)
    if (not os.path.exists(ruta_N)):
        os.mkdir(ruta_N)
    outputp=ruta_N+"/"+"fmdop0.pdf"
    plt.savefig(outputp)
    

#    ewc = np.repeat(ewc, 2)
    
    markers = ['v', '^'] * (len(ewoc) // 2)  # Lista alternada de marcadores
    colors = ['blue', 'red'] * (len(ewoc) // 2)  # Lista alternada de colores
    
    din = 0.7
    dout = (4 * M -1 - 2 * M * din) / (2 * M - 1)
    d = [1]
    pos = d[0]
    for i in range (2 * N -1):
        if i % 2 == 0:
            pos += din
            d.append(pos)
        else:
            pos += dout
            d.append(pos)
    
#    print("d", d)

    print("l: ", len(ewoc))     
    for i in range(len(ewoc)):
        plt.scatter(d[i], ewoc[i], marker=markers[i], color=colors[0], label="Eigenenergies w/o correlation" if i == 0 else None)
        plt.scatter(d[i], ewc[i], marker=markers[i], color=colors[1], label="Eigenenergies w correlation (FM)" if i == 0 else None)
#        print("d",d[i])
#        print("ewoc",ewoc[i])
#        print("ewc",ewc[i])
    plt.title(str(N) + "-atom chain" + ", $t_1$=" + str(round(t2, 2)) + ", $t_2$=" + str(round(t1, 2)) + ", U/$t_1$=" + str(U) + ", $\delta$=" + str(dop), fontsize = 16)
    plt.ylabel("E/$t_1$", fontsize = 15)
    plt.xlabel("State index", fontsize = 15)
    plt.grid(visible=True, lw=0.5)
    plt.legend(fontsize = 15)
    
    inset_ax = inset_axes(plt.gca(), width="30%", height="30%", loc='lower right')
    inset_ax.set_title("Edge states zoom", fontsize=12)
        # Customize the position and size of the inset plot as per your requirement
    inset_ax.set_xlim(18, 23)  # Set the x-axis limits for the inset plot
    inset_ax.set_ylim(-0.5, 2.1)  # Set the y-axis limits for the inset plot
    inset_ax.set_xticklabels([])  # Set the x-axis tick positions for the inset plot
    inset_ax.set_yticks([-0.5, 0.8, 2.1])  # Set the y-axis tick positions for the inset plot
# Plot the data for the inset plot
    for i in range(len(ewoc)):
        inset_ax.scatter(d[i], ewoc[i], marker= markers[i], color='blue', label="Eigenenergies w/o correlation")
        inset_ax.scatter(d[i], ewc[i], marker= markers[i], color='red', label="Eigenenergies w correlation (FM)")
        
    if (not os.path.exists(ruta_N)):
        os.mkdir(ruta_N)
    outputp=ruta_N+"/"+"fmdop2.png"
    plt.savefig(outputp)





def mupm (U,t1,t2,N,dop,T):
    b = 1/(kb*T)
    _,allenergies,_,_ = find_roots(M,D)
    print("\n ALL ENERGIES: ",allenergies)
    eq = allenergies[M] 
    print("eq", eq)
    _, ube, ue = correlation(U, t1, t2, N, dop)
    ube = np.sum(ube)
    print("\n(ep+em)/2 =", (ue*(2+dop)+2*ube)/2)
    
    def f(mu):
        ep = b*(eq-mu+ue*(1+dop/2)+ube)
        em = b*(-eq-mu+ue*(1+dop/2)+ube)
        return (1+dop/2)-(np.exp(ep)+1)**(-1)-(np.exp(em)+1)**(-1)
    
    emax = max(allenergies)
    
#    print("emax", emax) 
    mu = np.linspace(-emax, U*emax, 10000)
    plt.plot(mu,f(mu))
    plt.show()  
    musol = bisect(f, 0, U*emax, xtol=1e-20)
    print(musol)
    

def phsdgrm(U,t1,t2,dop):
    for i in range(2,41,0.1):
        if i<=2*Mc:
            y = 0 
        else: 
            ub, ube, UE = correlation(U,t1,t2,i,dop)
            enrgs, stts = eigen(t1,t2,i)
            enrgs = sorted(enrgs)
            print(" i", i)
            print("\nENERGÍAS: ", enrgs)
            eq = enrgs[i//2]/t1 
            print("eq = ", eq)   
#            dad                   
            UB = sum(el for a in ub for el in a)
            print("\nUB,", UB)
#            y = UE * (dop + dop**(2)/4 - 1) + UB + 2 * eq * (1 - dop)
            y = (2 * UE - eq)*dop + eq * (2 - dop) - UE * (1 + dop/2 )**2

            
        plt.scatter(i,y,color="red")
    plt.show()

def phsdgrm(U, t1, t2, dop):
    fm_points = []  # Puntos correspondientes a la fase FM
    pm_points = []  # Puntos correspondientes a la fase PM
    trivial_points = []  # Puntos correspondientes a la fase TRIVIAL

    U_values = np.arange(0.01, 1.6, 0.05)  # Valores de U desde 0.5 hasta 1.5 con incremento de 0.01

    for U_val in U_values:
        N = []  # Valores de i (N)
        for i in range(2, 81, 2):
            m = i // 2
            if i <= 2 * Mc:
                trivial_points.append((U_val, m))
            else:
                ub, ube, UE = correlation(U_val, t1, t2, i, dop)
                enrgs, stts = eigen(t1, t2, i)
                enrgs = sorted(enrgs)
                eq = enrgs[i // 2] / t1
                UB = sum(el for a in ub for el in a)
                y = (2 * UE - eq) * dop + eq * (2 - dop) - UE * (1 + dop / 2) ** 2
#                y = dop*(2*UE-eq)+eq*(2-dop)-UE*(1+dop/2)**2

                if y < 0:
                    fm_points.append((U_val, m))
                else:
                    pm_points.append((U_val, m))
            N.append(m)

    plt.scatter([point[1] for point in fm_points], [point[0] for point in fm_points], color='red', label='FM', marker= "s")
    plt.scatter([point[1] for point in pm_points], [point[0] for point in pm_points], color='blue', label='PM', marker ="s")
    plt.scatter([point[1] for point in trivial_points], [point[0] for point in trivial_points], color='green', label='TRIVIAL', marker = "s")
    plt.title("Magnetic phase diagram, " + "$t_1$=" + str(round(t2, 2)) + ", $t_2$=" + str(round(t1, 2)), fontsize=17)
    plt.xlabel("M", fontsize = 15)
    plt.ylabel("U/$t_1$", fontsize = 15)
    plt.legend(fontsize = 15)

    plt.xticks(np.arange(0, 41, 4))  # Establecer los ticks en incrementos de 4
    plt.axvline(x= Mc, color='black', linestyle='--')  # Línea vertical en Mc
    outputplot = ruta + "/"+ "phasediagramdop0.png"
    plt.savefig(outputplot)
    plt.show()  
 
            
        
            
#phsdgrm(U,t1,t2,dop)
    
#correlation(U,t1,t2,N,dop)   

#Aq = (0.5*(np.sinh(q*(2*M+1))/np.sinh(q)-1)-M)**(-1/2)


#print(hfinal(t1,t2,N))
#find_roots(M, D)

#correlation(3,t1,t2,N,dop)
#mupm(3,t1,t2,N,0,10)
#find_roots(M,D)
plotfm(U,t1,t2,N,dop)
#plotpm(U,t1,t2,N,dop)

#eigenstates(t1,t2,N)
#plot_energias(t1,t2,N)
#plot_all(t1,t2,N)         
