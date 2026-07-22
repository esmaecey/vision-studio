"""
Dataset Split worker — train/val/test bölme işlemini arka planda yapar.
Bölme sonunda çıktı klasörüne bir data.yaml üretir (pose ise kpt_shape/flip_idx dahil).

Kaynak klasör çözümlemesi ortak utils.dataset_paths modülünden gelir:
  - images -> labels YOLO kuralı,
  - zaten bölünmüş veri setlerinde (images/train, images/val ...) görselleri
    alt klasörlerden toplama.
"""
import os
import random
import shutil
import yaml
from PyQt5.QtCore import QThread, pyqtSignal

from utils.dataset_paths import (
    resolve_dataset_dirs, collect_dataset_items,
    keypoint_count_from_label_files, max_class_from_label_files,
)


class SplitWorker(QThread):
    progress = pyqtSignal(int)          # 0-100 ilerleme
    log = pyqtSignal(str)               # log mesajı
    finished_signal = pyqtSignal(dict)  # sonuç özeti

    def __init__(self, source_dir, output_dir, ratios, shuffle=True, seed=42,
                 class_names=None, flip_idx=None):
        super().__init__()
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.ratios = ratios            # (train, val, test) örn: (0.7, 0.2, 0.1)
        self.shuffle = shuffle
        self.seed = seed
        self.class_names = list(class_names) if class_names else []
        self.flip_idx = list(flip_idx) if flip_idx else []

    def run(self):
        try:
            images_dir, labels_dir = resolve_dataset_dirs(self.source_dir)
            self.log.emit(f"Görsel klasörü: {images_dir}")
            self.log.emit(f"Etiket klasörü: {labels_dir}")

            if not images_dir or not os.path.isdir(images_dir):
                self.log.emit(
                    f"HATA: Görsel klasörü bulunamadı: {images_dir}\n"
                    "Kaynak olarak görsellerin bulunduğu klasörü (örn. images/train) "
                    "ya da içinde images/ ve labels/ olan veri seti kökünü seçin."
                )
                self.finished_signal.emit({})
                return

            # Görselleri topla (doğrudan ya da train/val gibi alt klasörlerden)
            items, used_subdirs = collect_dataset_items(images_dir, labels_dir)
            if not items:
                self.log.emit(
                    "HATA: Hiç görsel bulunamadı — işlem durduruldu.\n"
                    f"Görsel klasörü: {images_dir}\n"
                    "Doğru klasörü seçtiğinizden emin olun (görseller doğrudan bu "
                    "klasörde ya da train/val gibi alt klasörlerde olmalı)."
                )
                self.finished_signal.emit({})
                return

            if used_subdirs:
                self.log.emit(
                    f"Bilgi: Veri seti zaten alt klasörlere ayrılmış "
                    f"({', '.join(used_subdirs)}). Tüm görseller ({len(items)}) "
                    "birleştirilip yeniden bölünüyor."
                )

            n_labeled = sum(1 for _, lp, _ in items if lp)
            self.log.emit(f"{len(items)} görsel bulundu, {n_labeled} tanesinde etiket var.")
            if n_labeled == 0:
                self.log.emit(
                    "UYARI: Hiçbir görselle eşleşen etiket bulunamadı. Etiketler "
                    f"beklenen konumda değil: {labels_dir}"
                )

            if self.shuffle:
                random.seed(self.seed)
                random.shuffle(items)
            else:
                items.sort(key=lambda x: x[2])

            n = len(items)
            train_r, val_r, test_r = self.ratios
            n_train = int(n * train_r)
            n_val = int(n * val_r)

            splits = {
                "train": items[:n_train],
                "val": items[n_train:n_train + n_val],
                "test": items[n_train + n_val:],
            }

            # Hedef klasör yapısını oluştur: train/images, train/labels, ...
            for split in splits:
                os.makedirs(os.path.join(self.output_dir, split, "images"), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, split, "labels"), exist_ok=True)

            total = n
            done = 0
            summary = {}

            for split, split_items in splits.items():
                summary[split] = len(split_items)
                for (img_path, label_path, out_name) in split_items:
                    # Görseli kopyala
                    dst_img = os.path.join(self.output_dir, split, "images", out_name)
                    shutil.copy(img_path, dst_img)

                    # Etiketi kopyala (varsa) — görselle aynı ada
                    if label_path and os.path.exists(label_path):
                        label_out = os.path.splitext(out_name)[0] + ".txt"
                        dst_label = os.path.join(self.output_dir, split, "labels", label_out)
                        shutil.copy(label_path, dst_label)

                    done += 1
                    self.progress.emit(int(done / total * 100))

                self.log.emit(f"{split}: {len(split_items)} görsel kopyalandı.")

            # --- data.yaml üret ---
            label_paths = [lp for _, lp, _ in items if lp]
            yaml_path = self._write_data_yaml(label_paths, splits)
            if yaml_path:
                summary["yaml"] = yaml_path
                self.log.emit(f"data.yaml oluşturuldu: {yaml_path}")

            self.log.emit("Bölme işlemi tamamlandı!")
            self.finished_signal.emit(summary)

        except Exception as e:
            self.log.emit(f"HATA: {str(e)}")
            self.finished_signal.emit({})

    def _write_data_yaml(self, label_paths, splits):
        """Bölünmüş veri seti için data.yaml üretir. Pose ise kpt_shape/flip_idx ekler."""
        # names: aktif projeden; yoksa etiketlerden nc tahmin edip genel isimler üret
        names = list(self.class_names)
        if not names:
            nc = max_class_from_label_files(label_paths) + 1
            if nc <= 0:
                self.log.emit("UYARI: Sınıf bilgisi yok; data.yaml 'names' alanı boş kalabilir. "
                              "Settings'ten data.yaml seçmeniz önerilir.")
                nc = 0
            else:
                self.log.emit("UYARI: Aktif proje yok; genel sınıf isimleri (class0..) yazıldı. "
                              "Doğru isimler için Settings'ten data.yaml seçin.")
            names = [f"class{i}" for i in range(nc)]

        data = {
            "path": os.path.abspath(self.output_dir),
            "train": "train/images",
            "val": "val/images",
        }
        if splits.get("test"):
            data["test"] = "test/images"
        data["nc"] = len(names)
        data["names"] = {i: n for i, n in enumerate(names)}

        # Pose veri seti: kpt_shape ve flip_idx ekle
        K = keypoint_count_from_label_files(label_paths)
        if K > 0:
            data["kpt_shape"] = [K, 3]
            if len(self.flip_idx) == K:
                data["flip_idx"] = list(self.flip_idx)
            else:
                # Varsayılan: kimlik eşlemesi (takas yok) — kullanıcı sonra düzenleyebilir
                data["flip_idx"] = list(range(K))
                self.log.emit("UYARI: Geçerli flip_idx bulunamadı; kimlik eşlemesi "
                              "(takas yok) yazıldı. Sol/sağ simetrik keypoint'ler için "
                              "data.yaml'daki flip_idx'i düzenleyin.")

        yaml_path = os.path.join(self.output_dir, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        return yaml_path
