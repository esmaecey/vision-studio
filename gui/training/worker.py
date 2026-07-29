"""
Training worker — YOLO Detect/Pose eğitimini arka planda çalıştırır.
Her epoch sonunda ilerleme ve metrik bilgisi yayar.
"""
import os
import time
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
        start_time = time.time()
        try:
            from ultralytics import YOLO

            c = self.config
            resume = bool(c.get("resume") and c.get("resume_model"))

            # Her epoch sonunda tetiklenecek callback
            def on_epoch_end(trainer):
                # Ultralytics, son epoch'tan SONRA bir kez daha final doğrulama için
                # bu callback'i tetikleyebilir; o satırı "Final doğrulama" olarak
                # ayır ki "Epoch 2/1" gibi yanıltıcı satır çıkmasın.
                total = getattr(trainer, "epochs", None) or self._total_epochs
                epoch = trainer.epoch + 1

                if epoch > total:
                    head = "Final doğrulama "
                    self.progress.emit(100)
                else:
                    head = f"Epoch {epoch}/{total} "
                    self.progress.emit(min(int(epoch / total * 100), 100))

                # Metrikleri topla (varsa). Pose eğitiminde Ultralytics iki ayrı set
                # üretir: box (B) ve pose/keypoint (P).
                metrics = trainer.metrics or {}
                box_map50 = metrics.get("metrics/mAP50(B)", 0.0)
                box_map5095 = metrics.get("metrics/mAP50-95(B)", 0.0)

                lines = [f"{head}| Box  mAP50: {box_map50:.4f}  mAP50-95: {box_map5095:.4f}"]

                # Keypoint (P) metrikleri yalnızca pose eğitiminde bulunur
                if "metrics/mAP50(P)" in metrics or "metrics/mAP50-95(P)" in metrics:
                    pose_map50 = metrics.get("metrics/mAP50(P)", 0.0)
                    pose_map5095 = metrics.get("metrics/mAP50-95(P)", 0.0)
                    pad = " " * len(head)
                    lines.append(f"{pad}| Pose mAP50: {pose_map50:.4f}  mAP50-95: {pose_map5095:.4f}")

                self.epoch_info.emit("\n".join(lines))

            if resume:
                self.log.emit(f"Kaldığı yerden devam ediliyor: {c['resume_model']}")
                model = YOLO(c["resume_model"])
                model.add_callback("on_fit_epoch_end", on_epoch_end)
                self.log.emit("Eğitim (resume) başlıyor...")
                model.train(resume=True)
            else:
                self.log.emit(f"Model yükleniyor: {c['model']}")
                model = YOLO(c["model"])
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
                    save_period=c.get("save_period", -1),
                )

            # Ultralytics'in gerçekte kullandığı çıktı klasörünü al
            result_dir = str(model.trainer.save_dir) if hasattr(model, "trainer") else os.path.join(c.get("project", ""), c.get("name", ""))
            duration = time.time() - start_time
            self.log.emit(f"Eğitim tamamlandı! Sonuçlar: {result_dir}")

            # Sonuçları özetleyen analiz.txt raporunu üret (hata olsa da eğitim biter)
            try:
                self._write_analysis_report(result_dir, c, duration)
            except Exception as e:
                self.log.emit(f"Analiz raporu yazılamadı: {e}")

            self.finished_signal.emit(result_dir)

        except Exception as e:
            self.log.emit(f"HATA: {str(e)}")
            self.finished_signal.emit("")

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_duration(seconds):
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}s {m}dk {s}sn"
        if m:
            return f"{m}dk {s}sn"
        return f"{s}sn"

    def _write_analysis_report(self, result_dir, c, duration):
        """
        Eğitim sonuçlarını insan tarafından okunabilir bir 'analiz.txt' olarak
        çıktı klasörüne yazar: model, veri seti, parametreler, süre, genel ve
        sınıf bazlı mAP skorları.
        """
        if not result_dir or not os.path.isdir(result_dir):
            return

        task = c.get("task", "detect")
        data_yaml = c.get("data")  # eğitimde kullanılan (mutlak) data.yaml
        best = os.path.join(result_dir, "weights", "best.pt")

        # --- Sınıf bazlı ve genel metrikleri best.pt üzerinde değerlendirerek al ---
        overall = {}
        per_class = []
        class_names = {}
        if os.path.exists(best) and data_yaml and os.path.exists(data_yaml):
            try:
                from ultralytics import YOLO
                self.log.emit("Analiz için son değerlendirme (val) çalıştırılıyor...")
                m = YOLO(best).val(data=data_yaml, split="val", verbose=False, plots=False)
                class_names = dict(getattr(m, "names", {}) or {})
                box = getattr(m, "box", None)
                if box is not None:
                    overall["Box"] = {
                        "mAP50": float(box.map50), "mAP50-95": float(box.map),
                        "Precision": float(box.mp), "Recall": float(box.mr),
                    }
                    maps = getattr(box, "maps", None)
                    if maps is not None:
                        for idx, name in class_names.items():
                            try:
                                per_class.append((int(idx), name, float(maps[int(idx)])))
                            except (IndexError, ValueError, TypeError):
                                continue
                pose = getattr(m, "pose", None)
                if pose is not None:
                    overall["Pose"] = {
                        "mAP50": float(pose.map50), "mAP50-95": float(pose.map),
                        "Precision": float(pose.mp), "Recall": float(pose.mr),
                    }
            except Exception as e:
                self.log.emit(f"Analiz değerlendirmesi atlandı: {e}")

        # --- Raporu oluştur ---
        lines = []
        lines.append("=" * 60)
        lines.append("VISION STUDIO — EĞİTİM ANALİZ RAPORU")
        lines.append("=" * 60)
        lines.append(f"Tarih                : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Görev (task)         : {task}")
        lines.append(f"Kullanılan model     : {c.get('resume_model') or c.get('model', '-')}")
        lines.append(f"Veri seti (data.yaml): {c.get('data_original') or data_yaml or '-'}")
        lines.append(f"Çıktı klasörü        : {result_dir}")
        lines.append(f"En iyi model         : {best if os.path.exists(best) else '-'}")
        lines.append(f"Eğitim süresi        : {self._fmt_duration(duration)}")
        lines.append("")
        lines.append("-" * 60)
        lines.append("EĞİTİM PARAMETRELERİ")
        lines.append("-" * 60)
        if c.get("resume"):
            lines.append("Mod                  : Kaldığı yerden devam (resume)")
        lines.append(f"Epochs               : {c.get('epochs', '-')}")
        lines.append(f"Image size           : {c.get('imgsz', '-')}")
        lines.append(f"Batch                : {c.get('batch', '-')}")
        lines.append(f"Device               : {c.get('device', '-')}")
        lines.append(f"Optimizer            : {c.get('optimizer', '-')}")
        lines.append(f"Learning rate (lr0)  : {c.get('lr0', '-')}")
        if c.get("save_period", -1) and c.get("save_period", -1) > 0:
            lines.append(f"Checkpoint (save_period): her {c['save_period']} epoch")
        lines.append("")

        if overall:
            lines.append("-" * 60)
            lines.append("GENEL SKORLAR")
            lines.append("-" * 60)
            for group, vals in overall.items():
                lines.append(f"[{group}]")
                for k, v in vals.items():
                    lines.append(f"  {k:<12}: {v:.4f}")
            lines.append("")

        if per_class:
            lines.append("-" * 60)
            lines.append("SINIF BAZLI mAP50-95")
            lines.append("-" * 60)
            lines.append(f"  {'ID':<4}{'Sınıf':<24}{'mAP50-95':>10}")
            for idx, name, ap in sorted(per_class, key=lambda x: x[0]):
                lines.append(f"  {idx:<4}{str(name):<24}{ap:>10.4f}")
            lines.append("")

        if not overall and not per_class:
            lines.append("Not: Sınıf bazlı/genel metrikler hesaplanamadı "
                         "(değerlendirme yapılamadı). results.csv dosyasını inceleyin.")
            lines.append("")

        lines.append("=" * 60)

        report_path = os.path.join(result_dir, "analiz.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.log.emit(f"Analiz raporu kaydedildi: {report_path}")