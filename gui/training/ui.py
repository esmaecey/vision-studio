"""
Eğitim Arayüzü — Detect ve Pose (Keypoint) eğitimi tek arayüzde.
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QProgressBar, QTextEdit, QGroupBox, QGridLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from .worker import TrainingWorker
from config.project_state import project_state


class TrainingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.data_yaml = None
        self.worker = None
        self.mode = "detect"  # detect veya pose
        self._build_ui()
        project_state.add_listener(self._on_project_changed)
        self._on_project_changed()

    def _build_ui(self):
        layout = QVBoxLayout()

        # --- Mod seçimi ---
        mode_layout = QHBoxLayout()
        self.btn_detect = QPushButton("Detect Training")
        self.btn_detect.setCheckable(True)
        self.btn_detect.setChecked(True)
        self.btn_detect.clicked.connect(lambda: self.set_mode("detect"))
        self.btn_pose = QPushButton("Keypoint (Pose) Training")
        self.btn_pose.setCheckable(True)
        self.btn_pose.clicked.connect(lambda: self.set_mode("pose"))
        mode_layout.addWidget(self.btn_detect)
        mode_layout.addWidget(self.btn_pose)
        layout.addLayout(mode_layout)

        # --- data.yaml seçimi ---
        data_group = QGroupBox("Veri Seti (data.yaml)")
        data_layout = QHBoxLayout()
        self.data_line = QLineEdit()
        self.data_line.setReadOnly(True)
        btn_data = QPushButton("data.yaml Seç")
        btn_data.clicked.connect(self.select_data)
        data_layout.addWidget(self.data_line)
        data_layout.addWidget(btn_data)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # --- Parametreler ---
        param_group = QGroupBox("Eğitim Parametreleri")
        grid = QGridLayout()

        grid.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self._update_model_options()
        grid.addWidget(self.model_combo, 0, 1)

        grid.addWidget(QLabel("Epochs:"), 0, 2)
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 1000)
        self.epoch_spin.setValue(50)
        grid.addWidget(self.epoch_spin, 0, 3)

        grid.addWidget(QLabel("Batch:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        grid.addWidget(self.batch_spin, 1, 1)

        grid.addWidget(QLabel("Image Size:"), 1, 2)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(128, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        grid.addWidget(self.imgsz_spin, 1, 3)

        grid.addWidget(QLabel("Device:"), 2, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "0"])  # 0 = ilk GPU
        grid.addWidget(self.device_combo, 2, 1)

        grid.addWidget(QLabel("Workers:"), 2, 2)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(4)
        grid.addWidget(self.workers_spin, 2, 3)

        grid.addWidget(QLabel("Learning Rate:"), 3, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(4)
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.01)
        grid.addWidget(self.lr_spin, 3, 1)

        grid.addWidget(QLabel("Optimizer:"), 3, 2)
        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["auto", "SGD", "Adam", "AdamW"])
        grid.addWidget(self.opt_combo, 3, 3)

        grid.addWidget(QLabel("Patience:"), 4, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 500)
        self.patience_spin.setValue(50)
        grid.addWidget(self.patience_spin, 4, 1)

        self.cache_check = QCheckBox("Cache (RAM)")
        grid.addWidget(self.cache_check, 4, 2)

        param_group.setLayout(grid)
        layout.addWidget(param_group)

        # --- Proje/İsim ---
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Proje Adı:"))
        self.project_line = QLineEdit("runs/train")
        name_layout.addWidget(self.project_line)
        name_layout.addWidget(QLabel("Deney Adı:"))
        self.name_line = QLineEdit("exp")
        name_layout.addWidget(self.name_line)
        layout.addLayout(name_layout)

        # --- Başlat ---
        self.btn_start = QPushButton("Eğitimi Başlat")
        self.btn_start.clicked.connect(self.start_training)
        layout.addWidget(self.btn_start)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # --- Log ve sonuç görseli yan yana ---
        bottom_layout = QHBoxLayout()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        bottom_layout.addWidget(self.log_area, stretch=1)

        self.result_image = QLabel("Eğitim sonucu grafikleri burada gösterilecek")
        self.result_image.setAlignment(Qt.AlignCenter)
        self.result_image.setMinimumWidth(400)
        self.result_image.setStyleSheet("border: 1px solid gray;")
        bottom_layout.addWidget(self.result_image, stretch=1)

        layout.addLayout(bottom_layout)
        self.setLayout(layout)

    def _update_model_options(self):
        self.model_combo.clear()
        if self.mode == "detect":
            self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
        else:
            self.model_combo.addItems(["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt"])

    def set_mode(self, mode):
        self.mode = mode
        self.btn_detect.setChecked(mode == "detect")
        self.btn_pose.setChecked(mode == "pose")
        self._update_model_options()

    def select_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "data.yaml Seç", "", "YAML (*.yaml)")
        if path:
            self.data_yaml = path
            self.data_line.setText(path)

    def _on_project_changed(self):
        """Merkezi projede data.yaml seçilmişse otomatik doldur."""
        if project_state.data_yaml_path and not self.data_yaml:
            self.data_yaml = project_state.data_yaml_path
            self.data_line.setText(project_state.data_yaml_path)

    def start_training(self):
        if not self.data_yaml:
            self.log_area.append("HATA: Önce data.yaml seçin.")
            return

        config = {
            "model": self.model_combo.currentText(),
            "data": self.data_yaml,
            "epochs": self.epoch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "batch": self.batch_spin.value(),
            "device": self.device_combo.currentText(),
            "workers": self.workers_spin.value(),
            "patience": self.patience_spin.value(),
            "cache": self.cache_check.isChecked(),
            "lr0": self.lr_spin.value(),
            "optimizer": self.opt_combo.currentText(),
            "project": self.project_line.text(),
            "name": self.name_line.text(),
        }

        self.btn_start.setEnabled(False)
        self.progress.setValue(0)
        self.log_area.clear()
        self.log_area.append(f"[{self.mode.upper()}] Eğitim hazırlanıyor...")

        self.worker = TrainingWorker(config)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.epoch_info.connect(self.log_area.append)
        self.worker.log.connect(self.log_area.append)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, result_dir):
        self.btn_start.setEnabled(True)

        if not result_dir:
            self.log_area.append("\nEğitim tamamlanamadı (hata oluştu).")
            return

        if not os.path.isdir(result_dir):
            self.log_area.append(f"\nUyarı: Sonuç klasörü bulunamadı: {result_dir}")
            return

        # results.png'yi göster (yoksa confusion matrix'i dene)
        for fname in ["results.png", "confusion_matrix.png"]:
            img_path = os.path.join(result_dir, fname)
            if os.path.exists(img_path):
                pixmap = QPixmap(img_path)
                self.result_image.setPixmap(
                    pixmap.scaled(self.result_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self.log_area.append(f"\nGrafik gösteriliyor: {fname}")
                break
        else:
            self.log_area.append("\nUyarı: Sonuç grafiği bulunamadı.")

        self.log_area.append(f"Eğitim çıktıları: {result_dir}")
        self.log_area.append(f"En iyi model: {os.path.join(result_dir, 'weights', 'best.pt')}")