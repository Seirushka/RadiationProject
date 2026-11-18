import numpy as np


def draw_integrand_ions(fig, x_common, y_total, section):
    """
    Рисует интеграл на переданном холсте (fig).

    :param fig: холст matplotlib Figure
    :param x_common: массив LET
    :param y_total: массив флюенса/спектра
    :param section: объект CrossSection с параметрами
    """
    fig.clf()
    ax1 = fig.add_subplot(111)

    sigma_values = section.sigma_ion(x_common)
    integrand = sigma_values * y_total * x_common * np.log(10.0)

    sigma_plot = np.copy(sigma_values)
    sigma_plot[sigma_plot == 0] = 1e-20

    integrand_plot = np.copy(integrand)
    integrand_plot[integrand_plot <= 1e-20] = 1e-20

    ax1.loglog(x_common, y_total, 'black', linewidth=2, label='Спектр (левая ось)')
    ax1.set_xlabel('LET, MeV/(g/cm$^2$)')
    ax1.set_ylabel('Спектр частиц', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, which='both', ls=':')

    ax2 = ax1.twinx()
    ax2.loglog(x_common, sigma_plot, 'red', linewidth=2, label='Сечение')
    ax2.loglog(x_common, integrand_plot, 'purple', linewidth=2.5, label='Произведение')
    ax2.fill_between(x_common, 1e-20, integrand_plot, color='purple', alpha=0.2, label='Площадь')

    ax2.set_ylabel('Сечение / Произведение', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    ax1.set_title(f'Интеграл (L0={section.L0})')
    fig.tight_layout()
