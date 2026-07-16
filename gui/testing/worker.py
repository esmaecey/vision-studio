"""
Testing worker — video/webcam akışında canlı YOLO inference yapar.
"""
import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class InferenceWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, float)  # işlenmiş kare, fps
    log = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, model_path, source, conf=0.25):
        super().__init__()
        self.model_path = model_path
        self.source = source          # 0 (webcam) veya dosya yolu
        self.conf = conf
        self._running = True

    def stop(self):
        self._running = False

    def set_conf(self, value):
        self.conf = value

    def run(self):
        try:
            from ultralytics import YOLO
            self.log.emit(f"Model yükleniyor: {self.model_path}")
            model = YOLO(self.model_path)

            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.log.emit("HATA: Kaynak açılamadı.")
                self.finished_signal.emit()
                return

            self.log.emit("Inference başladı.")

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    self.log.emit("Akış sona erdi.")
                    break

                start = time.time()
                results = model(frame, conf=self.conf, verbose=False)
                annotated = results[0].plot()  # kutuları/iskeleti çizer
                elapsed = time.time() - start
                fps = 1.0 / elapsed if elapsed > 0 else 0

                self.frame_ready.emit(annotated, fps)

            cap.release()
            self.finished_signal.emit()

        except Exception as e:
            self.log.emit(f"HATA: {str(e)}")
            self.finished_signal.emit()