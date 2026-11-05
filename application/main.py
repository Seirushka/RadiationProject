import matplotlib.pyplot as plt

from RadiationProject.application.calculators import combine_ion_spectrum, calculate_total_see
from RadiationProject.application.visualizer import plot_integrand_ions


def main():
    FILE_SEP = '../data/sl_x.txt'
    FILE_GCR = '../data/gl_x.txt'
    FILE_PROT = '../data/pp_x.txt'

    # Параметры (для теста)
    SIGMA_SH = 1e-8
    L0 = 1000.0
    SIGMA_P0 = 1e-12
    EP0 = 20.0
    DAYS = 365
    SHIELD_COL = 1

    print("="*40)
    print(" РАСЧЕТ ЧУВСТВИТЕЛЬНОСТИ ЭКБ К ОРЭ")
    print("="*40)

    results = calculate_total_see(
        FILE_SEP, FILE_GCR, FILE_PROT,
        sigma_sh=SIGMA_SH, L0=L0, 
        sigma_p0=SIGMA_P0, Ep0=EP0,
        days_in_mission=DAYS, col=SHIELD_COL
    )

    print(f"Сбоев от ионов:\t\t{results['mu_ion']:.4e}")
    print(f"Сбоев от протонов:\t{results['mu_prot']:.4e}")
    print(f"ИТОГО сбоев на бит:\t{results['mu_total']:.4e}")
    print(f"Вероятность сбоя:\t{results['probability']:.2f}%")
    print("="*40)

    print("Строим график...")
    x_let, y_total = combine_ion_spectrum(FILE_SEP, FILE_GCR, col=SHIELD_COL)
    fig = plot_integrand_ions(x_let, y_total, SIGMA_SH, L0)
    plt.show()

if __name__ == "__main__":
    main()
