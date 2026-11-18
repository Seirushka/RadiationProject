import matplotlib.pyplot as plt

from data_reader import load_data
from calculators import combine_ion_spectrum, calculate_total_see
#from visualizer import plot_integrand_ions
from physics_models import CrossSection

from gui import *

def main():
    FILE_SEP = 'data/sl_x.txt'
    FILE_GCR = 'data/gl_x.txt'
    FILE_PROT = 'data/pp_x.txt'

    # Параметры (для теста)
    SIGMA_SH = 1e-8
    L0 = 1000.0
    SIGMA_P0 = 1e-12
    EP0 = 20.0
    DAYS = 365
    SHIELD_COL = 1

    print("="*40)
    print(" ЗАГРУЗКА ДАННЫХ")
    print("="*40)
    data = load_data(FILE_SEP, FILE_GCR, FILE_PROT)

    errors = data.validate()
    if errors:
        print("Обнаружены проблемы:")
        for err in errors:
            print(f" - {err}")
        print("Продолжаем расчет...\n")
    else:
        print("Все файлы загружены корректно.\n")

    print("="*40)
    print(" РАСЧЕТ ЧУВСТВИТЕЛЬНОСТИ ЭКБ К ОРЭ")
    print("="*40)

    results = calculate_total_see(
        data,
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
    section_plot = CrossSection(sigma_sh=SIGMA_SH, L0=L0, k=10.0)
    x_let, y_total = combine_ion_spectrum(data, col=SHIELD_COL)
    fig = plot_integrand_ions(x_let, y_total, section_plot)
    plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
