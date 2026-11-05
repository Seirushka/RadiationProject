import numpy as np
import matplotlib.pyplot as plt

from RadiationProject.application.physics_models import sigma_ion


def plot_integrand_ions(x_common, y_total, sigma_sh, L0, k=10.0):
    sigma_values = sigma_ion(x_common, sigma_sh, L0, k)
    integrand = sigma_values * y_total * x_common * np.log(10.0)

    sigma_plot = np.copy(sigma_values)
    sigma_plot[sigma_plot == 0] = 1e-20

    integrand_plot = np.copy(integrand)
    integrand_plot[integrand_plot <= 1e-20] = 1e-20

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.loglog(x_common, y_total, 'black', linewidth=2, label='Спектр (левая ось)')
    ax1.set_xlabel('LET, MeV/(g/cm$^2$)')
    ax1.set_ylabel('Спектр частиц', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, which='both', ls=':')

    ax2 = ax1.twinx()
    ax2.loglog(x_common, sigma_plot, 'red', linewidth=2, label='Сечение (правая ось)')
    ax2.loglog(x_common, integrand_plot, 'purple', linewidth=2.5, label='Произведение')
    ax2.fill_between(x_common, 1e-20, integrand_plot, color='purple', alpha=0.2, label='Площадь')
    
    ax2.set_ylabel('Сечение / Произведение', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    plt.title(f'Интеграл (L0={L0})')
    plt.tight_layout()
    return fig
