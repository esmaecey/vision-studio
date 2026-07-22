"""
Training worker — YOLO Detect/Pose eğitimini arka planda çalıştırır.
Her epoch sonunda ilerleme ve metrik bilgisi yayar.
"""
import os
import tempfile
import yaml
from PyQt5.QtCore import QThread, pyqtSignal


def prepare_training_yaml(original_path):
    """
    data.yaml'ı eğitime hazırlar:
      1) 'path' alanını (göreliyse) yaml dosyasının klasörüne göre mutlak yapar.
         Ultralytics göreli 'path'i kendi varsayılan datasets dizinine göre
         çözdüğü için bu olmadan 'images not found' hatası alınır.
      2) train/val yollarının gerçekten var olup olmadığını doğrular.
      3) Orijinali DEĞİŞTİRMEZ; düzeltilmiş geçici bir kopya yazıp yolunu döner.

    Dönüş: (temp_yaml_path, warnings_list)
    Hata: FileNotFoundError / ValueError (kullanıcıya gösterilebilir mesajla).
    """
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"data.yaml bulunamadı:\n{original_path}")

    with open(original_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("data.yaml geçerli bir YAML sözlüğü değil.")

    yaml_dir = os.path.dirname(os.path.abspath(original_path))

    # 1) 'path' alanını mutlak hale getir (yoksa yaml'ın klasörü kök kabul edilir)
    root = data.get("path")
    if root:
        if not os.path.isabs(str(root)):
            root = os.path.normpath(os.path.join(yaml_dir, str(root)))
    else:
        root = yaml_dir
    data["path"] = root

    if not os.path.isdir(root):
        raise ValueError(
            f"Veri seti kök klasörü bulunamadı:\n{root}\n\n"
            "data.yaml içindeki 'path' alanını kontrol edin."
        )

    # 2) train/val/test yollarını çöz ve doğrula
    def resolve_split(value):
        items = value if isinstance(value, list) else [value]
        resolved = []
        for p in items:
            if not p:
                continue
            fp = str(p) if os.path.isabs(str(p)) else os.path.normpath(os.path.join(root, str(p)))
            resolved.append(fp)
        return resolved

    missing = []
    for split in ("train", "val"):
        if not data.get(split):
            missing.append(f"'{split}' alanı data.yaml'da tanımlı değil")
            continue
        for fp in resolve_split(data[split]):
            if not os.path.exists(fp):
                missing.append(f"{split}: {fp}")

    if missing:
        raise ValueError(
            "Eğitim başlatılamadı — veri yolları bulunamadı:\n  "
            + "\n  ".join(missing)
            + "\n\n'path' ve train/val klasörlerini kontrol edin."
        )

    warnings = []
    if data.get("test"):
        for fp in resolve_split(data["test"]):
            if not os.path.exists(fp):
                warnings.append(f"Uyarı: test yolu bulunamadı: {fp}")

    # 3) Düzeltilmiş geçici kopyayı yaz (tüm alanlar korunur: names, kpt_shape, flip_idx...)
    fd, temp_path = tempfile.mkstemp(prefix="visionstudio_data_", suffix=".yaml")
    os.close(fd)
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    return temp_path, warnings


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

                # Metrikleri topla (varsa). Pose eğitiminde Ultralytics iki ayrı set
                # üretir: box (B) ve pose/keypoint (P).
                metrics = trainer.metrics or {}
                box_map50 = metrics.get("metrics/mAP50(B)", 0.0)
                box_map5095 = metrics.get("metrics/mAP50-95(B)", 0.0)

                head = f"Epoch {epoch}/{self._total_epochs} "
                lines = [f"{head}| Box  mAP50: {box_map50:.4f}  mAP50-95: {box_map5095:.4f}"]

                # Keypoint (P) metrikleri yalnızca pose eğitiminde bulunur
                if "metrics/mAP50(P)" in metrics or "metrics/mAP50-95(P)" in metrics:
                    pose_map50 = metrics.get("metrics/mAP50(P)", 0.0)
                    pose_map5095 = metrics.get("metrics/mAP50-95(P)", 0.0)
                    pad = " " * len(head)
                    lines.append(f"{pad}| Pose mAP50: {pose_map50:.4f}  mAP50-95: {pose_map5095:.4f}")

                self.epoch_info.emit("\n".join(lines))

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
                exist_ok=c.get("exist_ok", False),
            )

            # Ultralytics'in gerçekte kullandığı çıktı klasörünü al
            result_dir = str(model.trainer.save_dir) if hasattr(model, "trainer") else os.path.join(c["project"], c["name"])
            self.log.emit(f"Eğitim tamamlandı! Sonuçlar: {result_dir}")
            self.finished_signal.emit(result_dir)

        except Exception as e:
            self.log.emit(f"HATA: {str(e)}")
            self.finished_signal.emit("")