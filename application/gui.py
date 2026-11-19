import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout,
    QWidget, QFileDialog, QMessageBox
)

from data_reader import SpectrumReader
from calculators import combine_ion_spectrum
from physics_models import CrossSection
from visualizer import draw_integrand_ions


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RadiationProject - Файлы не загружены")
        self.resize(800, 600)

        self.reader = SpectrumReader()
        self.col = 5

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        btn = QPushButton("Загрузить файл")
        btn.setFixedHeight(40)
        btn.clicked.connect(self.load_file)
        layout.addWidget(btn)

        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        self.update_plot()

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "*.txt")
        if not path:
            return
        try:
            self.reader.add_file(path)
            self.update_plot()
        except Exception as e:
            print(f"Ошибка загрузки:", str(e))

    def update_plot(self):
        data = self.reader.build()

        if data.sep is None and data.gcr is None:
            self.figure.clf()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Загрузите файлы",
                    ha='center', va='center', fontsize=20)
            ax.set_axis_off()
            self.canvas.draw()
            return

        errors = data.validate()
        if errors:
            QMessageBox.warning(self, "Внимание", "\n".join(errors))
            
            if not data.check_orbits():
                return

        section = CrossSection(sigma_sh=1e-8, L0=1.0,
                               sigma_p0=1e-12, Ep0=20.0)

        if data.sep is not None and data.gcr is not None:
            x, y = combine_ion_spectrum(data, col=self.col)
        elif data.sep is not None:
            x, y = data.sep.get_xy(self.col)
        else:
            x, y = data.gcr.get_xy(self.col)

        draw_integrand_ions(self.figure, x, y, section)
        self.canvas.draw()

        parts = []
        if data.sep: parts.append("СКЛ")
        if data.gcr: parts.append("ГКЛ")
        if data.protons: parts.append("Протоны")
        self.setWindowTitle(f"RadiationProject - Загружено: {', '.join(parts)}")
