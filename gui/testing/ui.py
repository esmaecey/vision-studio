"""
Test Arayüzü — eğitilmiş modeli görsel/video/webcam üzerinde çalıştırır.
"""
import os
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QFileDialog, QComboBox, QSlider, QTextEdit, QGroupBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from .worker import InferenceWorker, draw_skeleton_lines
from config.project_state import project_state
from gui.common.help import show_help

def imread_unicode(path):
    arr = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    ok, enc = cv2.imencode(ext, img)
    if ok:
        enc.tofile(path)
    return ok


class TestingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.model_path = None
        self.worker = None
        self.current_frame = None
        self._build_ui()
        project_state.add_listener(self._on_project_changed)
        self._on_project_changed()

    def _build_ui(self):
        layout = QVBoxLayout()

        # --- Model seçimi ---
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout()
        self.model_line = QLineEdit()
        self.model_line.setReadOnly(True)
        self.model_line.setPlaceholderText("best.pt veya yolov8n.pt seçin")
        btn_model = QPushButton("Model Seç (.pt)")
        btn_model.clicked.connect(self.select_model)
        model_layout.addWidget(self.model_line)
        model_layout.addWidget(btn_model)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # --- Kaynak seçimi ---
        source_group = QGroupBox("Kaynak")
        source_layout = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Görsel", "Video", "Webcam"])
        source_layout.addWidget(self.source_combo)

        self.btn_select_source = QPushButton("Dosya Seç")
        self.btn_select_source.clicked.connect(self.select_source_file)
        source_layout.addWidget(self.btn_select_source)

        self.source_line = QLineEdit()
        self.source_line.setReadOnly(True)
        source_layout.addWidget(self.source_line)
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # --- Confidence slider ---
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        conf_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel("0.25")
        conf_layout.addWidget(self.conf_label)
        layout.addLayout(conf_layout)

        # --- Kontrol butonları ---
        ctrl_layout = QHBoxLayout()
        self.btn_start = QPushButton("Başlat")
        self.btn_start.clicked.connect(self.start_inference)
        self.btn_stop = QPushButton("Durdur")
        self.btn_stop.clicked.connect(self.stop_inference)
        self.btn_stop.setEnabled(False)
        self.btn_snapshot = QPushButton("Snapshot Al")
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_snapshot)
        layout.addLayout(ctrl_layout)

        # --- Görüntü alanı ---
        self.display = QLabel("Sonuç burada gösterilecek")
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setMinimumHeight(400)
        self.display.setStyleSheet("border: 1px solid gray; background: #222;")
        layout.addWidget(self.display, stretch=1)

        # --- FPS ve log ---
        self.fps_label = QLabel("FPS: -")
        layout.addWidget(self.fps_label)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(100)
        layout.addWidget(self.log_area)

        self.setLayout(layout)

    def show_help(self):
        """F1 — modül bilgisi."""
        show_help(
            self,
            "Testing (Test) — Yardım",
            "Eğitilmiş bir modeli (.pt) görsel, video ya da webcam üzerinde çalıştırıp "
            "tespit/poz sonuçlarını canlı gösterir. Confidence eşiğini ayarlayabilir, "
            "sonucu snapshot olarak kaydedebilirsiniz.",
            [
                ("Kullanım", [
                    ("Model Seç (.pt)", "best.pt gibi eğitilmiş bir model seçer"),
                    ("Kaynak", "Görsel / Video / Webcam"),
                    ("Dosya Seç", "Görsel veya video dosyasını seçer"),
                    ("Confidence", "Tespit güven eşiği (düşük = daha çok tespit)"),
                    ("Başlat / Durdur", "Inference'ı başlatır / durdurur"),
                    ("Snapshot Al", "O anki kareyi görsel olarak kaydeder"),
                ]),
            ]
        )

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Model Seç", "", "PyTorch Model (*.pt)")
        if path:
            self.model_path = path
            self.model_line.setText(path)

    def _on_project_changed(self):
        """Settings'te varsayılan model seçilmişse otomatik doldur."""
        default_model = getattr(project_state, "default_model_path", None)
        if default_model and not self.model_path:
            self.model_path = default_model
            self.model_line.setText(default_model)

    def select_source_file(self):
        mode = self.source_combo.currentText()
        if mode == "Görsel":
            path, _ = QFileDialog.getOpenFileName(
                self, "Görsel Seç", "", "Görseller (*.jpg *.jpeg *.png *.bmp)"
            )
        elif mode == "Video":
            path, _ = QFileDialog.getOpenFileName(
                self, "Video Seç", "", "Videolar (*.mp4 *.avi *.mov *.mkv)"
            )
        else:
            self.log_area.append("Webcam seçildi, dosya seçmeye gerek yok.")
            return

        if path:
            self.source_line.setText(path)

    def on_conf_changed(self, value):
        conf = value / 100
        self.conf_label.setText(f"{conf:.2f}")
        if self.worker:
            self.worker.set_conf(conf)

    def start_inference(self):
        if not self.model_path:
            self.log_area.append("HATA: Önce model seçin.")
            return

        mode = self.source_combo.currentText()
        conf = self.conf_slider.value() / 100

        # Görsel modu: tek kare, thread'e gerek yok
        if mode == "Görsel":
            path = self.source_line.text()
            if not path:
                self.log_area.append("HATA: Görsel seçin.")
                return
            self._run_single_image(path, conf)
            return

        # Video / Webcam: thread ile
        if mode == "Video":
            source = self.source_line.text()
            if not source:
                self.log_area.append("HATA: Video seçin.")
                return
        else:
            source = 0  # webcam

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_area.clear()

        self.worker = InferenceWorker(self.model_path, source, conf,
                                      skeleton=list(project_state.skeleton))
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.log.connect(self.log_area.append)
        self.worker.finished_signal.connect(self.on_inference_finished)
        self.worker.start()

    def _run_single_image(self, path, conf):
        try:
            from ultralytics import YOLO
            self.log_area.append(f"Model yükleniyor: {os.path.basename(self.model_path)}")
            model = YOLO(self.model_path)

            img = imread_unicode(path)
            if img is None:
                self.log_area.append("HATA: Görsel okunamadı.")
                return

            results = model(img, conf=conf, verbose=False)
            annotated = results[0].plot(kpt_line=False)
            drew = draw_skeleton_lines(annotated, results[0], list(project_state.skeleton))
            if drew is False:
                self.log_area.append("Bilgi: Bu model için iskelet tanımı yok; sadece noktalar çiziliyor.")
            self.current_frame = annotated
            self._show_frame(annotated)

            n = len(results[0].boxes) if results[0].boxes is not None else 0
            self.log_area.append(f"Tespit edilen nesne sayısı: {n}")
            self.fps_label.setText("FPS: - (tek görsel)")

        except Exception as e:
            self.log_area.append(f"HATA: {str(e)}")

    def on_frame(self, frame, fps):
        self.current_frame = frame
        self._show_frame(frame)
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.display.setPixmap(
            pixmap.scaled(self.display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def stop_inference(self):
        if self.worker:
            self.worker.stop()

    def on_inference_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log_area.append("Durduruldu.")

    def take_snapshot(self):
        if self.current_frame is None:
            self.log_area.append("HATA: Kaydedilecek kare yok.")
            return

        # Kareyi tıklama anında dondur: dosya diyaloğu modal olduğundan açıkken
        # worker canlı önizlemeyi güncellemeye devam eder; kopya almazsak
        # kaydedilen kare, tıklama anındaki değil sonraki bir kare olur.
        frame = self.current_frame.copy()

        default_name = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        path, _ = QFileDialog.getSaveFileName(self, "Snapshot Kaydet", default_name, "JPEG (*.jpg)")
        if not path:
            return

        # Uzantı yoksa .jpg ekle (aksi halde imencode sessizce başarısız olur)
        if not os.path.splitext(path)[1]:
            path += ".jpg"

        if imwrite_unicode(path, frame):
            self.log_area.append(f"Snapshot kaydedildi: {os.path.abspath(path)}")
        else:
            self.log_area.append(f"HATA: Snapshot kaydedilemedi: {path}")
        # Canlı önizlemeye dokunmuyoruz; worker kesintisiz devam eder.