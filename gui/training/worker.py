"""
Training worker — YOLO Detect/Pose eğitimini arka planda çalıştırır.
Her epoch sonunda ilerleme ve metrik bilgisi yayar.
"""
import os
from PyQt5.QtCore import QThread, pyqtSignal


class TrainingWorker(QThread):
    progress = pyqtSignal(int)          # yüzde
    epoch_info = pyqtSignal(str)        # her epoch log satırı
    log = pyqtSignal(str)               # genel log
    finished_signal = pyqtSignal(str)   # eğitim sonucu klasör yolu

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._total_epochs = config.get("epochs", 50)

    def run(self):
        try:
            from ultralytics import YOLO

            c = self.config
            self.log.emit(f"Model yükleniyor: {c['model']}")
            model = YOLO(c["model"])

            # Her epoch sonunda tetiklenecek callback
            def on_epoch_end(trainer):
                epoch = trainer.epoch + 1
                pct = int(epoch / self._total_epochs * 100)
                self.progress.emit(pct)

                # Metrikleri topla (varsa)
                metrics = trainer.metrics or {}
                map50 = metrics.get("metrics/mAP50(B)", 0)
                map5095 = metrics.get("metrics/mAP50-95(B)", 0)
                self.epoch_info.emit(
                    f"Epoch {epoch}/{self._total_epochs} — mAP50: {map50:.4f}, mAP50-95: {map5095:.4f}"
                )

            model.add_callback("on_fit_epoch_end", on_epoch_end)

            self.log.emit("Eğitim başlıyor...")

            model.train(
                data=c["data"],
                epochs=c["epochs"],
                imgsz=c["imgsz"],
                batch=c["batch"],
                device=c["device"],
                workers=c["workers"],
                patience=c["patience"],
                cache=c["cache"],
                lr0=c["lr0"],
                optimizer=c["optimizer"],
                project=c["project"],
                name=c["name"],
                exist_ok=True,
            )

            # Ultralytics'in gerçekte kullandığı çıktı klasörünü al
            result_dir = str(model.trainer.save_dir) if hasattr(model, "trainer") else os.path.join(c["project"], c["name"])
            self.log.emit(f"Eğitim tamamlandı! Sonuçlar: {result_dir}")
            self.finished_signal.emit(result_dir)

        except Exception as e:
            self.log.emit(f"HATA: {str(e)}")
            self.finished_signal.emit("")