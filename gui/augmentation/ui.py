"""
Data Augmentation Arayüzü — veri artırma dönüşümlerini seçip uygular.
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QFileDialog, QCheckBox, QSpinBox, QProgressBar, QTextEdit, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt

from .worker import AugmentationWorker


class AugmentationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.images_dir = None
        self.labels_dir = None
        self.output_dir = None
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Kaynak seçimi
        src_group = QGroupBox("Kaynak (images/ ve labels/ içeren klasör)")
        src_layout = QHBoxLayout()
        self.src_line = QLineEdit()
        self.src_line.setReadOnly(True)
        btn_src = QPushButton("Kaynak Seç")
        btn_src.clicked.connect(self.select_source)
        src_layout.addWidget(self.src_line)
        src_layout.addWidget(btn_src)
        src_group.setLayout(src_layout)
        layout.addWidget(src_group)

        # Çıktı seçimi
        out_group = QGroupBox("Çıktı Klasörü")
        out_layout = QHBoxLayout()
        self.out_line = QLineEdit()
        self.out_line.setReadOnly(True)
        btn_out = QPushButton("Çıktı Seç")
        btn_out.clicked.connect(self.select_output)
        out_layout.addWidget(self.out_line)
        out_layout.addWidget(btn_out)
        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

        # Augmentation seçimi
        aug_group = QGroupBox("Augmentation Yöntemleri")
        aug_grid = QGridLayout()
        self.checks = {}
        options = [
            ("flip_h", "Yatay Çevir"), ("flip_v", "Dikey Çevir"),
            ("rotate", "Döndür"), ("brightness", "Parlaklık/Kontrast"),
            ("hue", "Renk Tonu"), ("blur", "Bulanıklık"),
            ("noise", "Gürültü"), ("clahe", "CLAHE"),
        ]
        for i, (key, label) in enumerate(options):
            cb = QCheckBox(label)
            self.checks[key] = cb
            aug_grid.addWidget(cb, i // 2, i % 2)
        # Varsayılan birkaçı seçili gelsin
        self.checks["flip_h"].setChecked(True)
        self.checks["rotate"].setChecked(True)
        self.checks["brightness"].setChecked(True)
        aug_group.setLayout(aug_grid)
        layout.addWidget(aug_group)

        # Görsel başına üretilecek sayı
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Her görselden kaç yeni üretilsin:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 20)
        self.count_spin.setValue(3)
        count_layout.addWidget(self.count_spin)
        count_layout.addStretch()
        layout.addLayout(count_layout)

        # Başlat
        self.btn_start = QPushButton("Augmentation Başlat")
        self.btn_start.clicked.connect(self.start_aug)
        layout.addWidget(self.btn_start)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        layout.addWidget(self.log_area)

        self.setLayout(layout)

    def select_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Kaynak Klasörü Seç")
        if folder:
            self.src_line.setText(folder)
            self.images_dir = os.path.join(folder, "images")
            self.labels_dir = os.path.join(folder, "labels")
            if not os.path.isdir(self.images_dir):
                # images/ alt klasörü yoksa, klasörün kendisini kullan
                self.images_dir = folder
                self.labels_dir = folder

    def select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Çıktı Klasörü Seç")
        if folder:
            self.out_line.setText(folder)
            self.output_dir = folder

    def start_aug(self):
        if not self.images_dir or not self.output_dir:
            self.log_area.append("HATA: Kaynak ve çıktı klasörü seçin.")
            return

        config = {key: cb.isChecked() for key, cb in self.checks.items()}
        if not any(config.values()):
            self.log_area.append("HATA: En az bir augmentation seçin.")
            return

        self.btn_start.setEnabled(False)
        self.progress.setValue(0)
        self.log_area.clear()
        self.log_area.append("Augmentation başlatılıyor...")

        self.worker = AugmentationWorker(
            self.images_dir, self.labels_dir, self.output_dir,
            config, self.count_spin.value()
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log_area.append)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, count):
        self.btn_start.setEnabled(True)
        self.log_area.append(f"\nİşlem bitti. Toplam {count} yeni görsel üretildi.")