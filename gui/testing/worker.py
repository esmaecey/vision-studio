"""
Testing worker — video/webcam akışında canlı YOLO inference yapar.
"""
import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


def draw_skeleton_lines(frame, result, skeleton, conf_thres=0.5,
                        color=(0, 255, 0), thickness=2):
    """
    Pose sonucundaki keypoint'leri, verilen skeleton (index çiftleri listesi) ile
    birleştirerek iskelet çizgilerini frame üzerine çizer.

    - Görünürlük/güven (v/conf) değeri conf_thres altındaki noktaların bağlantısı
      çizilmez (düşük görünürlüklü keypoint'ler bağlanmaz).
    - Skeleton index'i model çıktısındaki keypoint sayısını aşarsa o bağlantı atlanır.

    Dönüş:
      None  -> sonuç pose değil / keypoint yok
      False -> pose ama iskelet tanımı yok (yalnızca noktalar var)
      True  -> iskelet çizildi
    """
    kpts = getattr(result, "keypoints", None)
    if kpts is None or kpts.data is None or len(kpts.data) == 0:
        return None  # pose modeli değil ya da keypoint yok

    if not skeleton:
        return False  # iskelet tanımı yok

    data = kpts.data.cpu().numpy()  # (n_instances, n_kpts, 2 veya 3)
    for person in data:
        n = person.shape[0]
        has_conf = person.shape[1] >= 3
        for pair in skeleton:
            i, j = int(pair[0]), int(pair[1])
            if i < 0 or j < 0 or i >= n or j >= n:
                continue
            xi, yi = float(person[i][0]), float(person[i][1])
            xj, yj = float(person[j][0]), float(person[j][1])
            ci = float(person[i][2]) if has_conf else 1.0
            cj = float(person[j][2]) if has_conf else 1.0
            if ci < conf_thres or cj < conf_thres:
                continue
            # (0,0) olarak gelen tespit edilememiş noktaları atla
            if (xi <= 0 and yi <= 0) or (xj <= 0 and yj <= 0):
                continue
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)),
                     color, thickness, cv2.LINE_AA)
    return True


class InferenceWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, float)  # işlenmiş kare, fps
    log = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, model_path, source, conf=0.25, skeleton=None):
        super().__init__()
        self.model_path = model_path
        self.source = source          # 0 (webcam) veya dosya yolu
        self.conf = conf
        self.skeleton = list(skeleton) if skeleton else []
        self._warned_no_skeleton = False
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
                # Ultralytics'in kendi (COCO'ya sabit) iskeletini kapat; iskeleti
                # aktif projenin skeleton tanımından kendimiz çiziyoruz.
                annotated = results[0].plot(kpt_line=False)
                drew = draw_skeleton_lines(annotated, results[0], self.skeleton)
                if drew is False and not self._warned_no_skeleton:
                    self.log.emit("Bilgi: Bu model için iskelet tanımı yok; sadece noktalar çiziliyor.")
                    self._warned_no_skeleton = True
                elapsed = time.time() - start
                fps = 1.0 / elapsed if elapsed > 0 else 0

                self.frame_ready.emit(annotated, fps)

            cap.release()
            self.finished_signal.emit()

        except Exception as e:
            self.log.emit(f"HATA: {str(e)}")
            self.finished_signal.emit()