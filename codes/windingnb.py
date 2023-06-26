import numpy as np
import matplotlib.pyplot as plt
import sys, os

parent_dir = os.path.dirname(os.getcwd())
if not os.path.exists(os.path.join(parent_dir, "plots_ssh")):
    os.mkdir(os.path.join(parent_dir, "plots_ssh"))
ruta = os.path.join(parent_dir, "plots_ssh")

k_values = np.linspace(-np.pi, np.pi, 1000)  # Valores de k desde -pi a pi

v_values = [1, 1.2,  1.2]  # Valores de v
w_values = [1.2, 1.2, 1]  # Valores de w

fig, axs = plt.subplots(1, 3, figsize=(11, 4))

# Variables para almacenar los valores máximos y mínimos de hy
min_hy = np.inf
max_hy = -np.inf

for i, (v, w) in enumerate(zip(v_values, w_values)):
    hx = v + w * np.cos(k_values)
    hy = w * np.sin(k_values)

    axs[i].plot(hx, hy)
    axs[i].set_xlabel('$h_x$', fontsize=15)

    # Añadir label del eje y solo en el primer subplot
    if i == 0:
        axs[i].set_ylabel('$h_y$', fontsize=15)

    # Calcular el valor de w/v
    ratio = w / v

    title_label = chr(ord('c') + i)
    axs[i].set_title(f'({title_label}) $\Delta$ = {ratio:.2f}', fontsize=18)
    axs[i].grid(True)

    # Marcar el punto (0, 0) en rojo
    axs[i].plot(0, 0, 'ro')

    # Marcar el punto (v, 0) con una cruz
    axs[i].plot(v, 0, 'kx')


    # Dibujar flecha en k=π/2
    idx_pi_2 = np.abs(k_values - np.pi/2).argmin()
    x_arrow_start_pi_2 = hx[idx_pi_2]
    y_arrow_start_pi_2 = hy[idx_pi_2]
    x_arrow_end_pi_2 = hx[idx_pi_2 + 1]
    y_arrow_end_pi_2 = hy[idx_pi_2 + 1]
    dx_pi_2 = x_arrow_end_pi_2 - x_arrow_start_pi_2
    dy_pi_2 = y_arrow_end_pi_2 - y_arrow_start_pi_2
    axs[i].arrow(x_arrow_start_pi_2, y_arrow_start_pi_2, dx_pi_2, dy_pi_2, head_width=0.08, head_length=0.08, fc='black', ec='black')

    # Dibujar flecha en k=-π/2
    idx_minus_pi_2 = np.abs(k_values + np.pi/2).argmin()
    x_arrow_start_minus_pi_2 = hx[idx_minus_pi_2]
    y_arrow_start_minus_pi_2 = hy[idx_minus_pi_2]
    x_arrow_end_minus_pi_2 = hx[idx_minus_pi_2 + 1]
    y_arrow_end_minus_pi_2 = hy[idx_minus_pi_2 + 1]
    dx_minus_pi_2 = x_arrow_end_minus_pi_2 - x_arrow_start_minus_pi_2
    dy_minus_pi_2 = y_arrow_end_minus_pi_2 - y_arrow_start_minus_pi_2
    axs[i].arrow(x_arrow_start_minus_pi_2, y_arrow_start_minus_pi_2, dx_minus_pi_2, dy_minus_pi_2, head_width=0.08, head_length=0.08, fc='black', ec='black')

    # Actualizar los valores máximos y mínimos de hy
    min_hy = min(min_hy, np.min(hy) - 0.2)
    max_hy = max(max_hy, np.max(hy) + 0.2)

    # Eliminar etiquetas del eje y excepto en el primer subplot
    if i != 0:
        axs[i].set_yticklabels([])

    # Establecer los mismos xticks y x labels en todos los subplots
    plt.xticks(np.arange(-1.5, 2.6, 0.5))

    # Ajustar límite del eje x en el tercer subplot
    if i == 2:
        axs[i].set_xlim(-0.5, np.max(hx) + 0.2)

# Ajustar los límites de y en todos los subplots
for ax in axs:
    ax.set_ylim(min_hy, 1.5)
    ax.set_yticks(np.arange(-1.5, 1.6, 0.5))

plt.tight_layout()

outputplot = ruta + "/windingnumber.png"
plt.savefig(outputplot)
plt.show()

