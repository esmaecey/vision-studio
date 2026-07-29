"""
Vision Studio — Ana uygulama.
Sol navigation panel + sağda modül içeriği (QStackedWidget).
"""
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QStackedWidget, QLabel, QGroupBox, QShortcut
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

from config.project_state import project_state
from config.app_settings import load_settings, save_setting
from gui.detect_labeling.ui import DetectLabelingWidget
from gui.keypoint_labeling.ui import KeypointLabelingWidget
from gui.augmentation.ui import AugmentationWidget
from gui.dataset_split.ui import DatasetSplitWidget
from gui.data_validation.ui import DataValidationWidget
from gui.training.ui import TrainingWidget
from gui.testing.ui import TestingWidget
from gui.dashboard.home import HomeWidget
from gui.dashboard.settings import SettingsWidget
from config.theme import LIGHT, DARK, build_stylesheet, sidebar_stylesheet, logo_stylesheet


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Studio")
        self.setGeometry(60, 60, 1400, 850)
        self.current_palette = LIGHT

        # Kalıcı ayarları geri yükle: aktif proje (data.yaml), model, çıktı klasörü.
        # Sayfalar oluşturulmadan ÖNCE yüklenir ki sınıf listeleri dolu gelsin.
        project_state.restore_from_settings()

        self._build_ui()

        # Kayıtlı temayı uygula (yoksa açık tema)
        saved_theme = load_settings().get("theme", "light")
        self.apply_theme(saved_theme)

        # F1: aktif modülün yardım penceresini aç
        QShortcut(QKeySequence("F1"), self, self._show_current_help)

    def _build_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Sol navigation panel ---
        nav_container = QWidget()
        nav_layout = QVBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        self.logo = QLabel("Vision Studio")
        self.logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        nav_layout.addWidget(self.logo)

        self.nav_list = QListWidget()

        menu_items = [
            "Home",
            "Detect Labeling (Nesne Etiketleme)",
            "Keypoint Labeling (Anahtar Nokta Etiketleme)",
            "Data Augmentation (Veri Artırma)",
            "Dataset Split (Veri Bölme)",
            "Data Validation (Veri Doğrulama)",
            "Training (Eğitim)",
            "Testing (Test)",
            "Settings",
            "About",
        ]
        for item in menu_items:
            self.nav_list.addItem(QListWidgetItem(item))

        self.nav_list.currentRowChanged.connect(self.on_nav_changed)
        nav_layout.addWidget(self.nav_list)

        nav_container.setLayout(nav_layout)
        nav_container.setFixedWidth(230)
        main_layout.addWidget(nav_container)

        # --- Sağ içerik alanı ---
        content_container = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(20, 20, 20, 20)

        self.stack = QStackedWidget()
        self.stack.addWidget(HomeWidget())                      # 0
        self.stack.addWidget(DetectLabelingWidget())            # 1
        self.stack.addWidget(KeypointLabelingWidget())          # 2
        self.stack.addWidget(AugmentationWidget())              # 3
        self.stack.addWidget(DatasetSplitWidget())              # 4
        self.stack.addWidget(DataValidationWidget())            # 5
        self.stack.addWidget(TrainingWidget())                  # 6
        self.stack.addWidget(TestingWidget())                   # 7
        self.stack.addWidget(SettingsWidget(main_window=self))  # 8
        self.stack.addWidget(self._build_about())               # 9

        content_layout.addWidget(self.stack)
        content_container.setLayout(content_layout)
        main_layout.addWidget(content_container, stretch=1)

        self.setLayout(main_layout)
        self.nav_list.setCurrentRow(0)

    def _build_about(self):
        w = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)

        title = QLabel("Hakkında")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)

        card = QGroupBox("Proje Bilgisi")
        card_layout = QVBoxLayout()
        text = QLabel(
            "Vision Studio\n\n"
            "Nesne tespiti ve insan poz analizi için geliştirilmiş\n"
            "uçtan uca bilgisayarlı görü platformu.\n\n"
            "Veri etiketleme, veri artırma, veri bölme, model eğitimi ve\n"
            "model testi süreçlerini tek bir arayüzde birleştirir.\n\n"
            "Teknolojiler: Python · PyQt5 · OpenCV · Ultralytics YOLOv8\n\n"
            "Geliştirici: Esma Ece Yılmaz\n"
            "Staj Projesi — 2026"
        )
        text.setStyleSheet("font-size: 13px;")
        card_layout.addWidget(text)
        card.setLayout(card_layout)
        layout.addWidget(card)

        layout.addStretch()
        w.setLayout(layout)
        return w

    def on_nav_changed(self, index):
        self.stack.setCurrentIndex(index)

    def _show_current_help(self):
        """F1: o an açık olan modülün yardım penceresini açar."""
        current = self.stack.currentWidget()
        if hasattr(current, "show_help") and callable(current.show_help):
            current.show_help()
        else:
            from gui.common.help import show_help
            show_help(
                self, "Vision Studio — Yardım",
                "Bu modül için özel bir kısayol rehberi tanımlı değil. "
                "Etiketleme modüllerinde (Detect / Keypoint) F1 tüm kısayolları gösterir.",
                []
            )

    def apply_theme(self, theme):
        palette = DARK if theme == "dark" else LIGHT
        self.current_palette = palette
        self.setStyleSheet(build_stylesheet(palette))
        self.nav_list.setStyleSheet(sidebar_stylesheet(palette))
        self.logo.setStyleSheet(logo_stylesheet(palette))
        # Temayı kalıcı yap
        save_setting("theme", "dark" if theme == "dark" else "light")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())