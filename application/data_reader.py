import numpy as np
import re
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List


# КОНТЕЙНЕРЫ ДЛЯ ДАННЫХ
@dataclass
class SpectrumMetadata:
    """Метаданные из шапки файла."""
    apogee_km: Optional[float] = None               # самая высокая точка орбиты
    perigee_km: Optional[float] = None              # самая низкая точка орбиты
    inclination_deg: Optional[float] = None         # угол наклона орбиты к экватору Земли
    perigee_argument_deg: Optional[float] = None    # угол, где спутник ближе всего к Земле
    flight_time_years: Optional[float] = None       # время полета миссии
    start_year: Optional[float] = None              # год начала солнечного цикла
    solar_activity: Optional[str] = None            # степень солнечной активности
    solar_parity: Optional[str] = None              # четность солнечного цикла
    probability: Optional[float] = None             # вероятность превышения потока


@dataclass
class SpectrumData:
    """
    Все, что мы знаем об одном спектре.
    """
    filepath: str                       # путь к файлу
    spectrum_type: str                  # вид спектра - SEP | GCR | RB_PROTON
    x_name: str                         # ось х - LET | Energy
    x_units: str                        # единицы оси х - MeV/(g/cm2) | MeV
    y_type: str                         # ось у вид потока - FLUENCE | FLUX
    y_units: str                        # единицы оси у
    time_unit: Optional[str]            # если флюкс, время - mission | day | second
    metadata: SpectrumMetadata          # метаданные
    shielding_g_cm2: Dict[int, float]   # {col_index: толщина_в_г/см2}
    data: np.ndarray                    # вся таблица чисел

    # свойства
    @property
    def is_flux(self) -> bool:
        return self.y_type == 'FLUX'

    @property
    def is_fluence(self) -> bool:
        return self.y_type == 'FLUENCE'

    @property
    def columns(self) -> List[int]:
        """Индексы колонок с данными."""
        return sorted(self.shielding_g_cm2.keys())

    @property
    def n_points(self) -> int:
        return len(self.data)

    # методы
    def get_xy(self, col: int) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает (x, y) для указанной колонки."""
        if col not in self.shielding_g_cm2:
            raise ValueError(
                f"Колонка {col} не найдена в {self.filepath}. "
                f"Доступные: {self.columns}"
            )
        return self.data[:, 0].copy(), self.data[:, col].copy()

    def as_fluence(self, col: int, mission_days: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает спектр в единицах флюенса.
        Если в файле уже флюенс - возвращает как есть.
        Если флюкс - домножает на время миссии.
        """
        x, y = self.get_xy(col)
        if self.is_fluence:
            return x, y

        if mission_days is None:
            if self.metadata.flight_time_years is None:
                raise ValueError("Не задано время миссии.")
            mission_days = self.metadata.flight_time_years * 365.25

        if self.time_unit == 'day':
            return x, y * mission_days
        elif self.time_unit == 'second':
            return x, y * mission_days * 86400.0
        raise ValueError(
                f"Неизвестная единица времени '{self.time_unit}' в {self.filepath}"
                )

    def __repr__(self):
        return (
            f"<{self.spectrum_type}: {self.x_name} [{self.x_units}], "
            f"{self.y_type} [{self.y_units}], "
            f"{len(self.columns)} колонок, {self.n_points} точек>"
        )


@dataclass
class Data:
    """
    Контейнер со всеми прочитанными спектрами.
    Основной класс данных.
    """
    sep: Optional[SpectrumData] = None
    gcr: Optional[SpectrumData] = None
    protons: Optional[SpectrumData] = None

    # проверки
    def check_orbits(self) -> bool:
        """Проверяет, что SEP и GCR сняты с одинаковой орбиты."""
        if self.sep is None or self.gcr is None:
            return True

        s, g = self.sep.metadata, self.gcr.metadata

        def close(a, b, tol=1.0):
            if a is None or b is None:
                return True
            return abs(a - b) < tol

        return (
            close(s.apogee_km, g.apogee_km) and
            close(s.perigee_km, g.perigee_km) and
            close(s.inclination_deg, g.inclination_deg)
        )

    def orbit_warning(self) -> Optional[str]:
        """Если орбиты не совпадают — возвращает строку с предупреждением."""
        if self.check_orbits():
            return None
        s, g = self.sep.metadata, self.gcr.metadata
        return (
            "ВНИМАНИЕ: орбиты SEP и ГКЛ не совпадают!\n"
            f"  СКЛ: {s.apogee_km} * {s.perigee_km} км, {s.inclination_deg} рад\n"
            f"  ГКЛ: {g.apogee_km} * {g.perigee_km} км, {g.inclination_deg} рад"
        )

    def get_common_columns(self) -> List[int]:
        """Колонки, которые есть и у SEP, и у GCR."""
        if self.sep is None or self.gcr is None:
            return []
        return sorted(set(self.sep.columns) & set(self.gcr.columns))

    def get_common_shielding_mg(self) -> List[float]:
        """Толщины экранировки в мг/см², общие для SEP и GCR."""
        cols = self.get_common_columns()
        if not cols or self.sep is None:
            return []
        return [self.sep.shielding_mg_cm2[c] for c in cols]

    # валидация
    def validate(self) -> List[str]:
        """Проверяет базовую корректность загруженных данных."""
        errors = []
        # if self.sep is None:
        #     errors.append("Не загружен спектр СКЛ (SEP)")
        # if self.gcr is None:
        #     errors.append("Не загружен спектр ГКЛ (GCR)")
        # if self.protons is None:
        #     errors.append("Не загружен спектр протонов (RB PROTON)")

        warn = self.orbit_warning()
        if warn:
            errors.append(warn)

        if not self.get_common_columns() and self.sep and self.gcr:
            errors.append("У спектров СКЛ и ГКЛ нет общих колонок экранировки")

        return errors

    def __repr__(self):
        parts = []
        for name, spec in (('SEP', self.sep), ('GCR', self.gcr), ('PROTONS', self.protons)):
            parts.append(f"{name}={'ok' if spec else 'no'}")
        return f"<Data: {', '.join(parts)}>"


# РИДДЕР
class SpectrumReader:
    """Читает файлы спектров и разбирает их метаданные."""

    TYPE_PATTERNS = [
        ('SEP', r'SEP\s+(LET|PROTON)\s+SPECTRUM'),
        ('GCR', r'GCR\s+(LET|PROTON)\s+SPECTRUM'),
        ('RB_PROTON', r'RB\s+(PROTON)\s+SPECTRUM'),
    ]

    def __init__(self):
        self._spectra: Dict[str, SpectrumData] = {}

    # внутренние методы
    @staticmethod
    def _find_data_start(lines: List[str]) -> int:
        """Ищет первую строку с числовыми данными."""
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    return i
                except ValueError:
                    continue
        raise ValueError("Не найдена первая строка с числовыми данными")

    def _detect_type(self, header: List[str], filepath: str) -> str:
        """Определяет тип спектра: SEP, GCR или RB_PROTON."""
        text = ''.join(header)
        for spec_type, pattern in self.TYPE_PATTERNS:
            if re.search(pattern, text):
                return spec_type
        raise ValueError(
            f"Не удалось распознать тип спектра в {filepath}. "
            f"Ожидается одна из строк: SEP/GCR/RB ... SPECTRUM"
        )

    @staticmethod
    def _parse_metadata(header: List[str]) -> SpectrumMetadata:
        """Извлекает Apogee, Perigee, Inclination, Flight time и т.д."""
        text = ''.join(header)
        meta = SpectrumMetadata()

        m = re.search(r'Apogee\s*=\s*([\d.]+)\s*km', text, re.IGNORECASE)
        if m: meta.apogee_km = float(m.group(1))

        m = re.search(r'Perigee\s*=\s*([\d.]+)\s*km', text, re.IGNORECASE)
        if m: meta.perigee_km = float(m.group(1))

        m = re.search(r'Inclination\s*=\s*([\d.]+)\s*deg', text, re.IGNORECASE)
        if m: meta.inclination_deg = float(m.group(1))

        m = re.search(r'Argument\s+of\s+perigee\s*=\s*([\d.]+)\s*deg', text, re.IGNORECASE)
        if m: meta.perigee_argument_deg = float(m.group(1))

        m = re.search(r'Flight\s+time\s*=\s*([\d.]+)\s*year', text, re.IGNORECASE)
        if m: meta.flight_time_years = float(m.group(1))

        m = re.search(r'Start(?:\s+year[^=]*)?\s*=\s*([\d.]+)', text, re.IGNORECASE)
        if m: meta.start_year = float(m.group(1))

        m = re.search(r'Probability\s*=\s*([\d.]+)', text, re.IGNORECASE)
        if m: meta.probability = float(m.group(1))

        m = re.search(r'(?:Activity\s+of\s+)?Solar\s+cycle\s*[-=]\s*(\w+)', text, re.IGNORECASE)
        if m: meta.solar_activity = m.group(1).lower()

        m = re.search(r'Solar\s+cycle\s+number\s*[-=]\s*(\w+)', text, re.IGNORECASE)
        if m: meta.solar_parity = m.group(1).lower()

        return meta

    @staticmethod
    def _parse_x_axis(header: List[str], filepath: str) -> Tuple[str, str]:
        """Возвращает (имя_оси, единицы): ('LET', 'MeV/(g/cm2)') или ('Energy', 'MeV')."""
        for line in header:
            m = re.match(
                r'1\s*col\.\s*-\s*(LET|Energy)\s*\[([^\]]+)\]',
                line, re.IGNORECASE
            )
            if m:
                return m.group(1), m.group(2).strip()
        raise ValueError(f"В {filepath} не найдено описание оси X (LET или Energy)")

    @staticmethod
    def _parse_y_axis(header: List[str], filepath: str) -> Tuple[str, str, Optional[str]]:
        """
        Возвращает (y_type, y_units, time_unit):
        y_type = 'FLUENCE' | 'FLUX'
        time_unit = 'mission' | 'day' | 'second' | None
        """
        for line in header:
            m = re.search(
                r'(FLUENCE|FLUX)\s*\[([^\]]+)\]',
                line, re.IGNORECASE
            )
            if not m:
                continue

            y_type = m.group(1).upper()
            units = m.group(2).strip()

            if y_type == 'FLUENCE':
                return 'FLUENCE', units, 'mission'

            if re.search(r'\bday\b', units, re.IGNORECASE):
                time_unit = 'day'
            elif re.search(r'\b(?:s|sec)\b', units, re.IGNORECASE):
                time_unit = 'second'
            else:
                time_unit = None

            return 'FLUX', units, time_unit

        raise ValueError(f"В {filepath} не найдено описание оси Y (FLUENCE или FLUX)")

    @staticmethod
    def _parse_shielding(header: List[str], filepath: str) -> Dict[int, float]:
        """Возвращает {индекс_колонки: толщина(в г/см2}."""
        shielding = {}
        for line in header:
            m = re.match(
                r'(\d+)\s*col\.\s*-\s*shielding\s*=\s*([\d.Ee+-]+)',
                line, re.IGNORECASE
            )
            if m:
                col_num = int(m.group(1))
                if col_num < 2:
                    raise ValueError(f"Некорректный номер колонки защиты ({col_num}) в {filepath}")
                col_index = col_num - 1
                shielding[col_index] = float(m.group(2))
        
        if not shielding:
            raise ValueError(f"В {filepath} не найдены колонки экранировки")
        return shielding

    # публичные методы
    def add_file(self, filepath: str) -> SpectrumData:
        """Читает файл и добавляет его в коллекцию."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        data_start = self._find_data_start(lines)
        header = lines[:data_start]

        spectrum_type = self._detect_type(header, filepath)
        metadata = self._parse_metadata(header)
        x_name, x_units = self._parse_x_axis(header, filepath)
        y_type, y_units, time_unit = self._parse_y_axis(header, filepath)
        shielding = self._parse_shielding(header, filepath)

        data = np.loadtxt(lines[data_start:], dtype=float)

        if x_name == 'LET':
            data[:, 0] /= 1000.0
            for col in shielding.keys():
                data[:, col] *= 1000.0

        spectrum = SpectrumData(
            filepath=filepath,
            spectrum_type=spectrum_type,
            x_name=x_name,
            x_units=x_units.replace('/g/', '/mg/'),
            y_type=y_type,
            y_units=y_units.replace('/g/', '/mg/'),
            time_unit=time_unit,
            metadata=metadata,
            shielding_g_cm2=shielding,
            data=data,
        )

        if spectrum_type in self._spectra:
            old = self._spectra[spectrum_type].filepath
            print(
                f"ВНИМАНИЕ: спектр {spectrum_type} уже был загружен "
                f"из {old}. Заменяю на {filepath}."
            )

        self._spectra[spectrum_type] = spectrum
        return spectrum

    def build(self) -> Data:
        """Возвращает Data со всеми прочитанными спектрами."""
        return Data(
            sep=self._spectra.get('SEP'),
            gcr=self._spectra.get('GCR'),
            protons=self._spectra.get('RB_PROTON'),
        )
