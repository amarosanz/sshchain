"""
Este código representa la función de distribución de Fermi-Dirac a potencial químico fijo para distintas temperaturas

""" 
import numpy as np
import matplotlib.pyplot as plt

##Constantes
kb = 8.617333262e-5 #eV/K
deg = 1 #degeneración del estado
mu = 0.5 #potencial químico (eV)

e = np.linspace(0,2*mu,1000)
# Lista de 5 temperaturas
temperatures = [1, 75*3, 150*3, 225*3, 900]

fig, ax = plt.subplots()
for T in temperatures:
    beta = 1/(kb*T)
    fd = deg/(np.exp(beta*(e-mu))+1)

    ax.plot(e, fd, label='T = {} K'.format(T))

ax.set_xlabel('E')
ax.set_ylabel("Average occupation")
ax.set_title('Fermi-Dirac distribution for $\mu$ = {} eV'.format(mu))
ax.legend()

plt.show()

