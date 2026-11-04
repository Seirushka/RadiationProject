import numpy as np


def read_spectrum(filename, col=1):
    """
    Читает файл спектра, пропуская шапку в 20 строк.
    Возвращает два массива: LET (или Энергия) и Флюенс (или Поток).
    """
    try:
        data = np.loadtxt(filename, skiprows=20)
        x_data = data[:, 0]
        y_data = data[:, col]
        return x_data, y_data
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return np.array([]), np.array([])
