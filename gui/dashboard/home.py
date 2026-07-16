"""
Home ekranı — proje bilgisi, sistem durumu, son kullanılan model/dataset.
"""
import os
import platform
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QGridLayout
from PyQt5.QtCore import Qt


class HomeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(16)

        # Başlık
        title = QLabel("Vision Studio")
        title.setStyleSheet("font-size: 32px; font-weight: 700; padding-top: 10px;")
        layout.addWidget(title)

        subtitle = QLabel("Nesne tespiti ve poz analizi için uçtan uca bilgisayarlı görü platformu")
        subtitle.setStyleSheet("font-size: 14px; padding-bottom: 10px;")
        layout.addWidget(subtitle)

        # Modüller bilgisi
        modules_group = QGroupBox("Modüller")
        modules_layout = QVBoxLayout()
        modules_info = [
            "• Detect Etiketleme — YOLO formatında bounding box annotation",
            "• Keypoint Etiketleme — 17 noktalı insan iskeleti annotation",
            "• Data Augmentation — veri artırma ve etiket dönüşümü",
            "• Dataset Split — train/val/test bölme",
            "• Training — Detect ve Pose model eğitimi",
            "• Testing — görsel/video/webcam üzerinde canlı inference",
        ]
        for info in modules_info:
            lbl = QLabel(info)
            lbl.setTextFormat(Qt.PlainText)
            lbl.setStyleSheet("padding: 4px;")
            modules_layout.addWidget(lbl)
        modules_group.setLayout(modules_layout)
        layout.addWidget(modules_group)

        # Sistem durumu
        sys_group = QGroupBox("Sistem Durumu")
        sys_grid = QGridLayout()

        sys_grid.setVerticalSpacing(10)
        sys_grid.setColumnStretch(1, 1)

        sys_grid.addWidget(QLabel("İşletim Sistemi:"), 0, 0)
        os_label = QLabel(f"{platform.system()} {platform.release()}")
        os_label.setTextFormat(Qt.PlainText)          # ← ekle
        sys_grid.addWidget(os_label, 0, 1)

        sys_grid.addWidget(QLabel("Python:"), 1, 0)
        py_label = QLabel(platform.python_version())
        py_label.setTextFormat(Qt.PlainText)          # ← ekle
        sys_grid.addWidget(py_label, 1, 1)

        sys_grid.addWidget(QLabel("GPU Durumu:"), 2, 0)
        self.gpu_label = QLabel("Kontrol ediliyor...")
        sys_grid.addWidget(self.gpu_label, 2, 1)

        sys_group.setLayout(sys_grid)
        layout.addWidget(sys_group)

        layout.addStretch()
        self.setLayout(layout)

        self._check_gpu()

    def _check_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                self.gpu_label.setText(f"✅ GPU mevcut: {name}")
                self.gpu_label.setStyleSheet("color: green;")
            else:
                self.gpu_label.setText("⚠️ GPU yok — CPU kullanılacak")
                self.gpu_label.setStyleSheet("color: orange;")
        except Exception:
            self.gpu_label.setText("Torch yüklenemedi")