import numpy as np
import matplotlib.pyplot as plt

def plot_complex_angle(t1, t2):
    # Definir el rango de valores para k
    k_values = np.linspace(-np.pi, np.pi, 1000)
    
    # Crear una lista para almacenar los ángulos polares
    angles = []
    angles2 = []
    
    # Calcular el ángulo polar para cada valor de k
    for k_val in k_values:
        # Definir el número complejo dependiente de k
        z = t1 + t2 * np.exp(1j * k_val)
        z2 = t2 + t1 * np.exp(1j * k_val)
        
        # Calcular el ángulo polar y añadirlo a la lista
        angle = np.angle(z)
        angle2 = np.angle(z2)
        angles.append(angle)
        angles2.append(angle2)
    
    # Plotear la evolución del ángulo polar
    plt.plot(k_values, angles, label=f"$t_{1}$={t1}, $t_{2}$={t2}")
    plt.plot(k_values, angles2, label=f"$t_{1}$={t2}, $t_{2}$={t1}", color="red")
    plt.xlabel('k', fontsize=17)
    plt.ylabel('$θ_{k}$', fontsize=15)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['-π', '-π/2', '0', 'π/2', 'π'])
    plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['-π', '-π/2', '0', 'π/2', 'π'])
    plt.legend(fontsize=17)
    plt.grid()
    plt.savefig("polarangle.pdf")
    plt.show()

plot_complex_angle(1, 1.2)

