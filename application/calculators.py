import numpy as np

from data_reader import Data
from interpolation import pow_interp
from physics_models import CrossSection


class Integrator:
    """
    Класс для численного интегрирования и расчета частоты ОРЭ.
    """
    def __init__(self, use_simpson=True):
        """
        :param use_simpson: Если True — используем метод Симпсона, 
                            если False — метод трапеций.
        """
        self.use_simpson = use_simpson


    def __trapezoid(self, y, x):
        """
        Метод трапеций для интегрирования. (работает с любым шагом)
        """
        if len(x) < 2:
            return 0.0
        
        integral = 0.0
        for i in range(1, len(x)):
            dx = x[i] - x[i-1]
            height = (y[i] + y[i-1]) / 2.0
            integral += dx * height
        return integral


    def __simpson(self, y, x):
        """
        Метод Симпсона для интегрирования. (нужен равномерный шаг)
        Если точек чет колво, последний кусочек считаем методом трапеций.
        """
        n = len(x) - 1
        if n < 2:
            return self.__trapezoid(y, x)

        integral = 0.0
        limit = n if n % 2 == 0 else n - 1
        h = (x[1] - x[0])

        for i in range(0, limit, 2):
            integral += (h / 3.0) * (y[i] + 4*y[i+1] + y[i+2])
        if limit < n:
            integral += 0.5 * (y[limit] + y[n]) * (x[n] - x[limit])
            
        return integral


    def _integrate(self, y, x):
        """Выбор способа интегрирования."""
        if self.use_simpson:
            return self.__simpson(y, x)
        else:
            return self.__trapezoid(y, x)

        
    def calculate_ion(self, x_let, y_fluence, section:CrossSection):
        """Считает интеграл для ионов в лог-пространстве."""
        mask = x_let >= section.L0
        if not np.any(mask):
            return 0.0

        x_valid = x_let[mask]
        y_valid = y_fluence[mask]

        sigma_values = section.sigma_ion(x_valid)
        f_x = sigma_values * y_valid

        integrand = f_x * x_valid * np.log(10.0)
        log_x = np.log10(x_valid)

        return self._integrate(integrand, log_x)


    def calculate_protons(self, x_E, y_flux, section:CrossSection, days_in_mission=365):
        """Считает интеграл для протонов в лог-пространстве."""
        mask = x_E >= section.Ep0
        if not np.any(mask):
            return 0.0

        E_valid = x_E[mask]
        flux_valid = y_flux[mask]

        sigma_values = section.sigma_proton(E_valid)
        f_E = sigma_values * flux_valid

        integrand = f_E * E_valid * np.log(10.0)
        log_E = np.log10(E_valid)

        integral_value = self._integrate(y=integrand, x=log_E)
        return integral_value * days_in_mission


def combine_ion_spectrum(data: Data, col: int):
    """Объединяет СКЛ и ГКЛ в один общий спектр ионов."""
    if data.sep is None or data.gcr is None:
        raise ValueError("Для объединения нужны оба спектра: SEP и GCR")

    x_sep, y_sep = data.sep.get_xy(col)
    x_gcr, y_gcr = data.gcr.get_xy(col)

    mask_sep = (x_sep > 0) & (y_sep > 0)
    mask_gcr = (x_gcr > 0) & (y_gcr > 0)
    
    if not np.any(mask_sep) or not np.any(mask_gcr):
        return np.array([]), np.array([])

    x_min = min(x_sep[mask_sep].min(), x_gcr[mask_gcr].min())
    x_max = max(x_sep[mask_sep].max(), x_gcr[mask_gcr].max())

    x_common = np.logspace(np.log10(x_min), np.log10(x_max), 1000)

    y_sep_i = pow_interp(x_sep, y_sep, x_common, extrapolate=False)
    y_gcr_i = pow_interp(x_gcr, y_gcr, x_common, extrapolate=False)
    
    y_total = y_sep_i + y_gcr_i
    return x_common, y_total


def calculate_total_see(data: Data, 
                        sigma_sh, L0, sigma_p0, Ep0, 
                        days_in_mission=365, col=1):
    """
    Главная функция для интерфейса. Возвращает словарь с результатами.
    """
    section = CrossSection(
        sigma_sh=sigma_sh, L0=L0, k=10.0, 
        sigma_p0=sigma_p0, Ep0=Ep0
    )

    integrator = Integrator(use_simpson=True) 

    # ионы (СКЛ + ГКЛ)
    x_let, y_total = combine_ion_spectrum(data, col)
    mu_ion = integrator.calculate_ion(x_let, y_total, section)
    
    # протоны
    if data.protons is not None:
        x_p, y_p = data.protons.get_xy(col)
        mu_prot = integrator.calculate_protons(x_p, y_p, section, days_in_mission)
    else:
        mu_prot = 0.0

    # итог
    mu_total = mu_ion + mu_prot
    probability = (1.0 - np.exp(-mu_total)) * 100
    
    return {
        "mu_ion": mu_ion,
        "mu_prot": mu_prot,
        "mu_total": mu_total,
        "probability": probability
    }
