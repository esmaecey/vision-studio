"""
Dataset Split Arayüzü — veri setini train/val/test olarak böler.
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QFileDialog, QSpinBox, QCheckBox, QProgressBar, QTextEdit, QGroupBox
)
from PyQt5.QtCore import Qt

from .worker import SplitWorker


class DatasetSplitWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.source_dir = None
        self.output_dir = None
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # --- Kaynak klasör seçimi ---
        source_group = QGroupBox("Kaynak Veri Seti")
        source_layout = QVBoxLayout()
        source_row = QHBoxLayout()
        self.source_label = QLineEdit()
        self.source_label.setPlaceholderText("images/ ve labels/ içeren klasörü seçin")
        self.source_label.setReadOnly(True)
        btn_source = QPushButton("Kaynak Seç")
        btn_source.clicked.connect(self.select_source)
        source_row.addWidget(self.source_label)
        source_row.addWidget(btn_source)
        source_layout.addLayout(source_row)
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # --- Çıktı klasörü seçimi ---
        output_group = QGroupBox("Çıktı Klasörü")
        output_layout = QHBoxLayout()
        self.output_label = QLineEdit()
        self.output_label.setPlaceholderText("Bölünmüş veri setinin kaydedileceği klasör")
        self.output_label.setReadOnly(True)
        btn_output = QPushButton("Çıktı Seç")
        btn_output.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(btn_output)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # --- Oran ayarları ---
        ratio_group = QGroupBox("Bölme Oranları (%)")
        ratio_layout = QHBoxLayout()

        ratio_layout.addWidget(QLabel("Train:"))
        self.train_spin = QSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(70)
        self.train_spin.valueChanged.connect(self._update_ratio_sum)
        ratio_layout.addWidget(self.train_spin)

        ratio_layout.addWidget(QLabel("Val:"))
        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 100)
        self.val_spin.setValue(20)
        self.val_spin.valueChanged.connect(self._update_ratio_sum)
        ratio_layout.addWidget(self.val_spin)

        ratio_layout.addWidget(QLabel("Test:"))
        self.test_spin = QSpinBox()
        self.test_spin.setRange(0, 100)
        self.test_spin.setValue(10)
        self.test_spin.valueChanged.connect(self._update_ratio_sum)
        ratio_layout.addWidget(self.test_spin)

        ratio_group.setLayout(ratio_layout)
        layout.addWidget(ratio_group)

        self.sum_label = QLabel("Toplam: 100%")
        layout.addWidget(self.sum_label)

        # --- Seçenekler ---
        options_layout = QHBoxLayout()
        self.shuffle_check = QCheckBox("Karıştır (Shuffle)")
        self.shuffle_check.setChecked(True)
        options_layout.addWidget(self.shuffle_check)

        options_layout.addWidget(QLabel("Random Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(42)
        options_layout.addWidget(self.seed_spin)
        options_layout.addStretch()
        layout.addLayout(options_layout)

        # --- Başlat butonu ---
        self.btn_start = QPushButton("Bölmeyi Başlat")
        self.btn_start.clicked.connect(self.start_split)
        layout.addWidget(self.btn_start)

        # --- İlerleme ve log ---
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        layout.addWidget(self.log_area)

        self.setLayout(layout)

    def _update_ratio_sum(self):
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        self.sum_label.setText(f"Toplam: {total}%")
        if total == 100:
            self.sum_label.setStyleSheet("color: green;")
        else:
            self.sum_label.setStyleSheet("color: red;")

    def select_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Kaynak Klasörü Seç")
        if folder:
            self.source_dir = folder
            self.source_label.setText(folder)

    def select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Çıktı Klasörü Seç")
        if folder:
            self.output_dir = folder
            self.output_label.setText(folder)

    def start_split(self):
        if not self.source_dir:
            self.log_area.append("HATA: Önce kaynak klasör seçin.")
            return
        if not self.output_dir:
            self.log_area.append("HATA: Önce çıktı klasörü seçin.")
            return

        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        if total != 100:
            self.log_area.append("HATA: Oranların toplamı 100 olmalı.")
            return

        ratios = (
            self.train_spin.value() / 100,
            self.val_spin.value() / 100,
            self.test_spin.value() / 100,
        )

        self.btn_start.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_area.clear()
        self.log_area.append("Bölme işlemi başlatılıyor...")

        self.worker = SplitWorker(
            self.source_dir, self.output_dir, ratios,
            shuffle=self.shuffle_check.isChecked(),
            seed=self.seed_spin.value()
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.log_area.append)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, summary):
        self.btn_start.setEnabled(True)
        if summary:
            self.log_area.append(
                f"\nÖzet → Train: {summary.get('train', 0)}, "
                f"Val: {summary.get('val', 0)}, Test: {summary.get('test', 0)}"
            )