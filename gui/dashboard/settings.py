"""
Settings ekranı — aktif proje (data.yaml), tema, varsayılan yollar.
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QLineEdit, QFileDialog, QGroupBox, QMessageBox, QListWidget
)
from PyQt5.QtCore import Qt

from config.project_state import project_state


class SettingsWidget(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._build_ui()
        project_state.add_listener(self._on_project_changed)

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(14)

        title = QLabel("Ayarlar")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)

        # ---------- AKTİF PROJE ----------
        project_group = QGroupBox("Aktif Proje")
        project_layout = QVBoxLayout()

        desc = QLabel(
            "Seçilen data.yaml dosyasındaki sınıflar tüm etiketleme, eğitim ve "
            "test modüllerinde kullanılır."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px;")
        project_layout.addWidget(desc)

        yaml_row = QHBoxLayout()
        self.yaml_line = QLineEdit()
        self.yaml_line.setReadOnly(True)
        self.yaml_line.setPlaceholderText("Henüz bir data.yaml seçilmedi")
        yaml_row.addWidget(self.yaml_line)
        btn_yaml = QPushButton("data.yaml Seç")
        btn_yaml.clicked.connect(self.select_data_yaml)
        yaml_row.addWidget(btn_yaml)
        project_layout.addLayout(yaml_row)

        self.status_label = QLabel(project_state.summary())
        self.status_label.setStyleSheet("font-size: 12px; padding-top: 4px;")
        project_layout.addWidget(self.status_label)

        project_layout.addWidget(QLabel("Sınıflar:"))
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(160)
        project_layout.addWidget(self.class_list)

        project_group.setLayout(project_layout)
        layout.addWidget(project_group)

        # ---------- GÖRÜNÜM ----------
        theme_group = QGroupBox("Görünüm")
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Tema:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Açık (Light)", "Koyu (Dark)"])
        self.theme_combo.currentIndexChanged.connect(self.on_theme_changed)
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch()
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        # ---------- VARSAYILAN YOLLAR ----------
        paths_group = QGroupBox("Varsayılan Yollar")
        paths_layout = QVBoxLayout()

        m_row = QHBoxLayout()
        m_row.addWidget(QLabel("Model (.pt):"))
        self.model_line = QLineEdit()
        self.model_line.setPlaceholderText("Test ve eğitim için varsayılan model")
        m_row.addWidget(self.model_line)
        btn_m = QPushButton("Seç")
        btn_m.clicked.connect(self.select_model_path)
        m_row.addWidget(btn_m)
        paths_layout.addLayout(m_row)

        # Eğitim çıktı klasörü (opsiyonel; boşsa data.yaml klasörü kullanılır)
        o_row = QHBoxLayout()
        o_row.addWidget(QLabel("Çıktı Klasörü:"))
        self.output_line = QLineEdit()
        self.output_line.setReadOnly(True)
        self.output_line.setPlaceholderText("Varsayılan: data.yaml'ın bulunduğu klasör")
        o_row.addWidget(self.output_line)
        btn_o = QPushButton("Seç")
        btn_o.clicked.connect(self.select_output_dir)
        o_row.addWidget(btn_o)
        btn_o_clear = QPushButton("Temizle")
        btn_o_clear.clicked.connect(self.clear_output_dir)
        o_row.addWidget(btn_o_clear)
        paths_layout.addLayout(o_row)

        out_desc = QLabel(
            "Eğitim çıktıları buraya '/runs/<görev>/<deney adı>' altında kaydedilir. "
            "Boş bırakılırsa data.yaml'ın bulunduğu klasör kullanılır."
        )
        out_desc.setWordWrap(True)
        out_desc.setStyleSheet("font-size: 11px; color: gray;")
        paths_layout.addWidget(out_desc)

        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)

        layout.addStretch()
        self.setLayout(layout)

        self._refresh_project_info()

    # ---------- Eylemler ----------
    def select_data_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "data.yaml Seç", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            classes = project_state.load_data_yaml(path)
            QMessageBox.information(
                self, "Proje Yüklendi",
                f"{len(classes)} sınıf yüklendi.\n"
                f"Keypoint sayısı: {project_state.num_keypoints()}\n\n"
                "Bu proje tüm modüllerde aktif olarak kullanılacak."
            )
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"data.yaml okunamadı:\n{str(e)}")

    def select_model_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Varsayılan Model", "", "PyTorch (*.pt)")
        if path:
            self.model_line.setText(path)
            project_state.default_model_path = path

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Çıktı Klasörü Seç", "")
        if path:
            project_state.set_output_dir(path)
            self.output_line.setText(path)

    def clear_output_dir(self):
        project_state.set_output_dir(None)
        self.output_line.clear()

    def on_theme_changed(self, index):
        if self.main_window:
            self.main_window.apply_theme("dark" if index == 1 else "light")

    # ---------- Güncellemeler ----------
    def _on_project_changed(self):
        self._refresh_project_info()

    def _refresh_project_info(self):
        if project_state.data_yaml_path:
            self.yaml_line.setText(project_state.data_yaml_path)
        self.status_label.setText(project_state.summary())

        self.class_list.clear()
        for i, name in enumerate(project_state.class_names):
            self.class_list.addItem(f"{i}: {name}")