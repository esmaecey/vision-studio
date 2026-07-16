"""
Dataset Split worker — train/val/test bölme işlemini arka planda yapar.
"""
import os
import random
import shutil
from PyQt5.QtCore import QThread, pyqtSignal


class SplitWorker(QThread):
    progress = pyqtSignal(int)          # 0-100 ilerleme
    log = pyqtSignal(str)               # log mesajı
    finished_signal = pyqtSignal(dict)  # sonuç özeti

    def __init__(self, source_dir, output_dir, ratios, shuffle=True, seed=42):
        super().__init__()
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.ratios = ratios            # (train, val, test) örn: (0.7, 0.2, 0.1)
        self.shuffle = shuffle
        self.seed = seed

    def run(self):
        try:
            images_dir = os.path.join(self.source_dir, "images")
            labels_dir = os.path.join(self.source_dir, "labels")

            if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
                self.log.emit("HATA: Kaynak klasörde 'images' ve 'labels' alt klasörleri bulunamadı.")
                self.finished_signal.emit({})
                return

            valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
            images = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_ext)]

            if not images:
                self.log.emit("HATA: images klasöründe görsel bulunamadı.")
                self.finished_signal.emit({})
                return

            if self.shuffle:
                random.seed(self.seed)
                random.shuffle(images)

            n = len(images)
            train_r, val_r, test_r = self.ratios
            n_train = int(n * train_r)
            n_val = int(n * val_r)

            splits = {
                "train": images[:n_train],
                "val": images[n_train:n_train + n_val],
                "test": images[n_train + n_val:],
            }

            # Hedef klasör yapısını oluştur: train/images, train/labels, ...
            for split in splits:
                os.makedirs(os.path.join(self.output_dir, split, "images"), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, split, "labels"), exist_ok=True)

            total = n
            done = 0
            summary = {}

            for split, files in splits.items():
                summary[split] = len(files)
                for img_file in files:
                    base = os.path.splitext(img_file)[0]
                    label_file = base + ".txt"

                    # Görseli kopyala
                    src_img = os.path.join(images_dir, img_file)
                    dst_img = os.path.join(self.output_dir, split, "images", img_file)
                    shutil.copy(src_img, dst_img)

                    # Etiketi kopyala (varsa)
                    src_label = os.path.join(labels_dir, label_file)
                    if os.path.exists(src_label):
                        dst_label = os.path.join(self.output_dir, split, "labels", label_file)
                        shutil.copy(src_label, dst_label)

                    done += 1
                    self.progress.emit(int(done / total * 100))

                self.log.emit(f"{split}: {len(files)} görsel kopyalandı.")

            self.log.emit("Bölme işlemi tamamlandı!")
            self.finished_signal.emit(summary)

        except Exception as e:
            self.log.emit(f"HATA: {str(e)}")
            self.finished_signal.emit({})