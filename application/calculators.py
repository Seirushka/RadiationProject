import numpy as np
from scipy.integrate import simpson # заменить на самописный

from RadiationProject.application.data_reader import read_spectrum
from RadiationProject.application.interpolation import pow_interp
from RadiationProject.application.physics_models import sigma_ion, sigma_proton


def combine_ion_spectrum(file_sep, file_gcr, col=1):
    """Объединяет СКЛ и ГКЛ в один общий спектр ионов."""
    x_sep, y_sep = read_spectrum(file_sep, col)
    x_gcr, y_gcr = read_spectrum(file_gcr, col)

    # Ищем границы только по ненулевым данным
    mask_sep = (x_sep > 0) & (y_sep > 0)
    mask_gcr = (x_gcr > 0) & (y_gcr > 0)
    
    if not np.any(mask_sep) or not np.any(mask_gcr):
        return np.array([]), np.array([])

    x_min = max(x_sep[mask_sep].min(), x_gcr[mask_gcr].min())
    x_max = min(x_sep[mask_sep].max(), x_gcr[mask_gcr].max())

    x_common = np.logspace(np.log10(x_min), np.log10(x_max), 1000)

    y_sep_i = pow_interp(x_sep, y_sep, x_common, extrapolate=False)
    y_gcr_i = pow_interp(x_gcr, y_gcr, x_common, extrapolate=False)
    
    y_total = y_sep_i + y_gcr_i
    return x_common, y_total


def calculate_ion(x_let, y_fluence, sigma_sh, L0, k=10.0):
    """Считает интеграл для ионов в лог-пространстве."""
    mask = x_let >= L0
    if not np.any(mask):
        return 0.0

    x_valid = x_let[mask]
    y_valid = y_fluence[mask]

    sigma_values = sigma_ion(x_valid, sigma_sh, L0, k)
    f_x = sigma_values * y_valid

    # Интегрирование в лог-пространстве: dx = x * ln(10) * d(log10(x))
    integrand = f_x * x_valid * np.log(10.0)
    log_x = np.log10(x_valid)
    
    return simpson(y=integrand, x=log_x)


def calculate_protons(x_E, y_flux, sigma_p0, Ep0, days_in_mission=365):
    """Считает интеграл для протонов в лог-пространстве."""
    mask = x_E >= Ep0
    if not np.any(mask):
        return 0.0

    E_valid = x_E[mask]
    flux_valid = y_flux[mask]

    sigma_values = sigma_proton(E_valid, sigma_p0, Ep0)
    f_E = sigma_values * flux_valid

    integrand = f_E * E_valid * np.log(10.0)
    log_E = np.log10(E_valid)
    
    integral_value = simpson(y=integrand, x=log_E)
    return integral_value * days_in_mission


def calculate_total_see(file_sep, file_gcr, file_prot, 
                        sigma_sh, L0, sigma_p0, Ep0, 
                        days_in_mission=365, col=1):
    """Главная функция для интерфейса. Возвращает словарь с результатами."""

    # 1. Ионы (СКЛ + ГКЛ)
    x_let, y_total = combine_ion_spectrum(file_sep, file_gcr, col)
    mu_ion = calculate_ion(x_let, y_total, sigma_sh, L0)
    
    # 2. Протоны
    x_p, y_p = read_spectrum(file_prot, col)
    mu_prot = calculate_protons(x_p, y_p, sigma_p0, Ep0, days_in_mission)
    
    # 3. Итог
    mu_total = mu_ion + mu_prot
    probability = (1.0 - np.exp(-mu_total)) * 100 # в процентах
    
    return {
        "mu_ion": mu_ion,
        "mu_prot": mu_prot,
        "mu_total": mu_total,
        "probability": probability
    }
