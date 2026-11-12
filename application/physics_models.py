import numpy as np


class CrossSection:
    """
    Класс, инкапсулирующий параметры и формулы сечений ОРЭ.
    """
    def __init__(self, sigma_sh, L0, k=10.0, sigma_p0=None, Ep0=None):
        """
        Инициализация параметров.
        
        :param sigma_sh: Сечение насыщения для ионов (см^2/бит)
        :param L0: Пороговая ЛПЭ для ионов (МэВ·см^2/г)
        :param k: Коэффициент в экспоненте (по умолчанию 10.0)
        :param sigma_p0: Сечение насыщения для протонов (см^2/бит)
        :param Ep0: Пороговая энергия протонов (МэВ)
        """
        self.sigma_sh = sigma_sh
        self.L0 = L0
        self.k = k
        self.sigma_p0 = sigma_p0
        self.Ep0 = Ep0


    def sigma_ion(self, L):
        """
        Формула для CASRAD - аппроксимация сечения ОРЭ по ионам.
        """
        L = np.asarray(L, dtype=float)
        result = np.zeros_like(L)
        mask = L >= self.L0

        if np.any(mask):
            result[mask] = self.sigma_sh * np.exp(-self.k * self.L0 / L[mask])
        
        return result


    def sigma_proton(self, E):
        """
        Формула Бендела - аппроксимация сечения ОРЭ по протонам.
        """
        E = np.asarray(E, dtype=float)
        result = np.zeros_like(E)
        mask = E >= self.Ep0

        if np.any(mask):
            part1 = np.sqrt(np.sqrt(18.0 / self.Ep0) * (E[mask] - self.Ep0))
            part2 = 1.0 - np.exp(-0.18 * part1)
            result[mask] = self.sigma_p0 * (part2 ** 4)

        return result
