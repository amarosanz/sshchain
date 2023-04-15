import sys, os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from scipy.optimize import fsolve

parent_dir = os.path.dirname(os.getcwd())
if (not os.path.exists(os.path.join(parent_dir, "plots_ssh"))):
    os.mkdir(os.path.join(parent_dir, "plots_ssh"))
ruta = os.path.join(parent_dir, "plots_ssh")
rutap = ruta + "/Infinite chain"





k=np.linspace(-np.pi,np.pi,1000)

def eplus(k,Delta):
    return(np.sqrt((1-Delta)**2+4*Delta*(np.cos(k/2))**2))

def emoins(k,Delta):
    return(-np.sqrt((1-Delta)**2+4*Delta*(np.cos(k/2))**2))

def plot(k):
    Delta_list=[1000,1.2/0.8,1,0.8/1.2,0]
    Deltar = []
    for num in Delta_list:
        numr=round(num,2)
        Deltar.append(numr)
    fig, axs = plt.subplots(nrows=1, ncols=len(Delta_list), figsize=(12,3))
    is_first_plot = True
    for i, Delta in enumerate(Delta_list):
        Deltap=eplus(k,Delta)/max(eplus(k,Delta))
        Deltam=emoins(k,Delta)/max(eplus(k,Delta))
    
        axs[i].plot(k,Deltap, label="$\epsilon_{k_+}$",color="blue")
        axs[i].plot(k,Deltam, label="$\epsilon_{k_-}$",color="blue")
        axs[i].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axs[i].set_xlabel(r'ka', fontsize=11)
        axs[i].set_xticks([-np.pi, 0, np.pi])
        axs[i].set_xticklabels(['-π', '0', 'π'])
        ticks = [-1, -0.5, 0, 0.5, 1]
        labels = ['-1', '', '0', '', '1']
        axs[i].set_yticks(ticks)
        axs[i].set_yticklabels(labels)
        axs[i].tick_params(axis='x', top=True, labeltop=False)
        axs[i].tick_params(axis='y', right=True, labelright=False)
        axs[i].set_title(f"$\Delta$={Deltar[i]}", fontsize=13)
        if is_first_plot:
            axs[i].set_ylabel(r'Normalized energy', fontsize=11)
#            axs[i].legend(fontsize=13)
            is_first_plot = False
    plt.tight_layout()
#    outputplot="SSHinfinite.pdf"
#    plt.savefig(outputplot)



plot(k)




#plt.plot(k,eplus(k,Delta), label="$\epsilon_{k_+}$")
#plt.plot(k,emoins(k,Delta), label="$\epsilon_{k_-}$")
#plt.ylabel(r'Energía', fontsize=11)
#plt.xlabel(r'ka', fontsize=11)
#plt.legend(fontsize=13)
#plt.title("Bandas cadena SSH, $\Delta$=1")

if (not os.path.exists(rutap)):
    os.mkdir(rutap)
outputplot=rutap+"/SSHinfinite.pdf"
plt.savefig(outputplot)





