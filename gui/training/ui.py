"""
Eğitim Arayüzü — Detect ve Pose (Keypoint) eğitimi tek arayüzde.
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QProgressBar, QTextEdit, QGroupBox, QGridLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QDesktopServices
from PyQt5.QtCore import Qt, QUrl

from .worker import TrainingWorker, prepare_training_yaml
from config.project_state import project_state


class TrainingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.data_yaml = None
        self.worker = None
        self.mode = "detect"  # detect veya pose
        self._temp_data_yaml = None  # eğitim için üretilen geçici data.yaml
        self._last_exp_name = None   # kullanıcının girdiği deney adı (hata sonrası korunur)
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

        # --- Deney adı ---
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Deney Adı:"))
        self.name_line = QLineEdit("exp")
        name_layout.addWidget(self.name_line)
        layout.addLayout(name_layout)

        # --- Çıktı klasörü (otomatik hesaplanır, salt-okunur) ---
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Çıktı Klasörü:"))
        self.project_line = QLineEdit()
        self.project_line.setReadOnly(True)
        self.project_line.setPlaceholderText("data.yaml seçilince otomatik belirlenir")
        out_layout.addWidget(self.project_line)
        layout.addLayout(out_layout)
        self.name_line.textChanged.connect(self._update_output_display)

        # --- Başlat ---
        self.btn_start = QPushButton("Eğitimi Başlat")
        self.btn_start.clicked.connect(self.start_training)
        layout.addWidget(self.btn_start)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # Eğitim bitince çıktı klasörünü dosya gezgininde açan buton
        self.btn_open_dir = QPushButton("Çıktı Klasörünü Aç")
        self.btn_open_dir.clicked.connect(self._open_result_dir)
        self.btn_open_dir.setEnabled(False)
        self.last_result_dir = None
        layout.addWidget(self.btn_open_dir)

        # --- Log ve sonuç görseli yan yana ---
        bottom_layout = QHBoxLayout()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        # Metrik satırlarının hizalı görünmesi için sabit genişlikli font
        self.log_area.setFontFamily("Consolas")
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
        self._update_output_display()

    def select_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "data.yaml Seç", "", "YAML (*.yaml)")
        if path:
            self.data_yaml = path
            self.data_line.setText(path)
            self._update_output_display()

    def _on_project_changed(self):
        """Merkezi projede data.yaml seçilmişse otomatik doldur."""
        if project_state.data_yaml_path and not self.data_yaml:
            self.data_yaml = project_state.data_yaml_path
            self.data_line.setText(project_state.data_yaml_path)
        self._update_output_display()

    def _compute_project_dir(self):
        """
        Eğitim çıktı klasörünü mutlak yol olarak hesaplar: {base}/runs/{görev}
        base = Settings'teki çıktı klasörü, yoksa seçilen data.yaml'ın klasörü.
        Ultralytics'e mutlak yol verildiğinden 'runs' ikilenmesi olmaz.
        """
        base = project_state.training_base_dir(fallback_data_yaml=self.data_yaml)
        if not base:
            return None
        return os.path.join(base, "runs", self.mode)

    def _update_output_display(self):
        proj = self._compute_project_dir()
        if proj:
            name = self.name_line.text().strip() or "exp"
            self.project_line.setText(os.path.join(proj, name))
        else:
            self.project_line.clear()

    def _open_result_dir(self):
        if self.last_result_dir and os.path.isdir(self.last_result_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_result_dir))

    def start_training(self):
        if not self.data_yaml:
            self.log_area.append("HATA: Önce data.yaml seçin.")
            return

        project_dir = self._compute_project_dir()
        if not project_dir:
            self.log_area.append("HATA: Çıktı klasörü belirlenemedi (data.yaml yolu geçersiz).")
            return

        exp_name = self.name_line.text().strip() or "exp"
        # Kullanıcının girdiği ham değeri sakla ki hata durumunda geri yükleyebilelim
        self._last_exp_name = self.name_line.text()

        # Aynı deney adı zaten varsa kullanıcıya sor: üzerine yaz / yeni klasör / iptal
        exist_ok = False
        target_dir = os.path.join(project_dir, exp_name)
        if os.path.isdir(target_dir):
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Deney Klasörü Mevcut")
            box.setText(
                f"Bu deney adı zaten var:\n{target_dir}\n\nNe yapmak istersiniz?"
            )
            overwrite_btn = box.addButton("Üzerine Yaz", QMessageBox.DestructiveRole)
            new_btn = box.addButton("Yeni Klasör Oluştur", QMessageBox.AcceptRole)
            cancel_btn = box.addButton("İptal", QMessageBox.RejectRole)
            box.exec_()
            clicked = box.clickedButton()
            if clicked == cancel_btn:
                self.log_area.append("Eğitim iptal edildi (deney adı mevcut).")
                return
            exist_ok = (clicked == overwrite_btn)

        # data.yaml'ı hazırla: 'path'i mutlak yap, train/val varlığını doğrula,
        # orijinali bozmadan geçici kopya üret. Hata varsa eğitimi başlatma.
        try:
            prepared_yaml, warnings = prepare_training_yaml(self.data_yaml)
        except (FileNotFoundError, ValueError) as e:
            self.log_area.append(f"HATA: {str(e)}")
            QMessageBox.critical(self, "Veri Seti Hatası", str(e))
            return
        for w in warnings:
            self.log_area.append(w)
        self._temp_data_yaml = prepared_yaml

        config = {
            "model": self.model_combo.currentText(),
            "data": prepared_yaml,
            "epochs": self.epoch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "batch": self.batch_spin.value(),
            "device": self.device_combo.currentText(),
            "workers": self.workers_spin.value(),
            "patience": self.patience_spin.value(),
            "cache": self.cache_check.isChecked(),
            "lr0": self.lr_spin.value(),
            "optimizer": self.opt_combo.currentText(),
            "project": project_dir,   # mutlak yol: {base}/runs/{görev}
            "name": exp_name,
            "task": self.mode,
            "exist_ok": exist_ok,     # True: üzerine yaz, False: otomatik yeni klasör
        }

        self.btn_open_dir.setEnabled(False)
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

        # Eğitim için üretilen geçici data.yaml'ı temizle
        if self._temp_data_yaml and os.path.exists(self._temp_data_yaml):
            try:
                os.remove(self._temp_data_yaml)
            except OSError:
                pass
        self._temp_data_yaml = None

        # Deney adı alanı herhangi bir sebeple boşaldıysa kullanıcının değerini koru
        if self._last_exp_name and not self.name_line.text().strip():
            self.name_line.setText(self._last_exp_name)

        if not result_dir:
            self.log_area.append("\nEğitim tamamlanamadı (hata oluştu).")
            return

        if not os.path.isdir(result_dir):
            self.log_area.append(f"\nUyarı: Sonuç klasörü bulunamadı: {result_dir}")
            return

        # Çıktı klasörünü sakla ve "Klasörü Aç" butonunu etkinleştir
        self.last_result_dir = os.path.abspath(result_dir)
        self.btn_open_dir.setEnabled(True)

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