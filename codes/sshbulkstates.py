import sys, os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from scipy.optimize import fsolve

parent_dir = os.path.dirname(os.getcwd())
if not os.path.exists(os.path.join(parent_dir, "plots_ssh")):
    os.mkdir(os.path.join(parent_dir, "plots_ssh"))
ruta = os.path.join(parent_dir, "plots_ssh")
rutap = ruta + "/Infinite chain"

k = np.linspace(-np.pi, np.pi, 1000)

def eplus(k, Delta):
    return np.sqrt((1 - Delta)**2 + 4 * Delta * (np.cos(k / 2))**2)

def emoins(k, Delta):
    return -np.sqrt((1 - Delta)**2 + 4 * Delta * (np.cos(k / 2))**2)

def plot(k):
    Delta_list = [1000, 1.2 / 1, 1, 1 / 1.2, 0]
    Deltar = []
    for num in Delta_list:
        numr = round(num, 2)
        Deltar.append(numr)
    
    fig, axs = plt.subplots(nrows=1, ncols=len(Delta_list), figsize=(12, 3))
    
    labels = ['a)', 'b)', 'c)', 'd)', 'e)']  # Labels for subplots
    
    for i, Delta in enumerate(Delta_list):
        Deltap = eplus(k, Delta) / max(eplus(k, Delta))
        Deltam = emoins(k, Delta) / max(eplus(k, Delta))
    
        axs[i].plot(k, Deltap, label="$\epsilon_{k_+}$", color="blue")
        axs[i].plot(k, Deltam, label="$\epsilon_{k_-}$", color="blue")
        axs[i].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axs[i].set_xlabel(r'ka', fontsize=15)
        axs[i].set_xticks([-np.pi, 0, np.pi])
        axs[i].set_xticklabels(['-π', '0', 'π'], fontsize = 13)
        axs[i].tick_params(axis='x', top=True, labeltop=False)
        axs[i].tick_params(axis='y', right=True, labelright=False)
        axs[i].set_title(f"{labels[i]} $\Delta$={Deltar[i]}", fontsize=18)  # Adding subplot labels in titles
    
        if i != 0:
            axs[i].set_yticklabels([])  # Remove y-axis labels
    
    axs[0].set_ylabel(r'Énergie normalisée', fontsize=15)  # Add label only to the first subplot
    axs[0].set_yticks([-1, 0, 1])  # Set the tick positions on the y-axis
    axs[0].set_yticklabels([1, 0, -1], fontsize=13)  # Set the tick labels with the desired values and fontsize

    
    plt.tight_layout()
    outputplot = rutap + "/SSHinfinite.png"
    plt.savefig(outputplot)

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





