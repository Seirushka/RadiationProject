import numpy as np


def sigma_ion(L, sigma_sh, L0, k=10.0):
    """
    Экспоненциальная аппроксимация сечения ОРЭ по ионам.
    sigma_z(L) = sigma_sh * exp(-k * L0 / L) при L >= L0, иначе 0.
    """
    L = np.asarray(L, dtype=float)
    result = np.zeros_like(L)
    mask = L >= L0
    result[mask] = sigma_sh * np.exp(-k * L0 / L[mask])
    return result


def sigma_proton(E, sigma_p0, Ep0):
    """
    Формула Бендела для вероятности сбоя от протонов.
    E - массив энергий
    sigma_p0 - вероятность сбоя
    Ep0 - порог энергии
    """
    E = np.asarray(E, dtype=float)
    result = np.zeros_like(E)
    mask = E >= Ep0

    if np.any(mask):
        part1 = np.sqrt(np.sqrt(18.0 / Ep0) * (E[mask] - Ep0))
        part2 = 1.0 - np.exp(-0.18 * part1)
        result[mask] = sigma_p0 * (part2 ** 4)

    return result