"""
Data Augmentation Arayüzü — veri artırma dönüşümlerini seçip uygular.
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QFileDialog, QCheckBox, QSpinBox, QProgressBar, QTextEdit, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt

from .worker import AugmentationWorker
from config.project_state import project_state
from utils.dataset_paths import (
    VALID_IMG_EXT, dir_has_images, resolve_dataset_dirs, keypoint_count_from_dir,
)


class AugmentationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.images_dir = None
        self.labels_dir = None
        self.output_dir = None
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Kaynak seçimi
        src_group = QGroupBox("Kaynak (images/ ve labels/ içeren klasör)")
        src_layout = QHBoxLayout()
        self.src_line = QLineEdit()
        self.src_line.setReadOnly(True)
        btn_src = QPushButton("Kaynak Seç")
        btn_src.clicked.connect(self.select_source)
        src_layout.addWidget(self.src_line)
        src_layout.addWidget(btn_src)
        src_group.setLayout(src_layout)
        layout.addWidget(src_group)

        # Çıktı seçimi
        out_group = QGroupBox("Çıktı Klasörü")
        out_layout = QHBoxLayout()
        self.out_line = QLineEdit()
        self.out_line.setReadOnly(True)
        btn_out = QPushButton("Çıktı Seç")
        btn_out.clicked.connect(self.select_output)
        out_layout.addWidget(self.out_line)
        out_layout.addWidget(btn_out)
        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

        # Augmentation seçimi
        aug_group = QGroupBox("Augmentation Yöntemleri")
        aug_grid = QGridLayout()
        self.checks = {}
        options = [
            ("flip_h", "Yatay Çevir"), ("flip_v", "Dikey Çevir"),
            ("rotate", "Döndür"), ("brightness", "Parlaklık/Kontrast"),
            ("hue", "Renk Tonu"), ("blur", "Bulanıklık"),
            ("noise", "Gürültü"), ("clahe", "CLAHE"),
        ]
        for i, (key, label) in enumerate(options):
            cb = QCheckBox(label)
            self.checks[key] = cb
            aug_grid.addWidget(cb, i // 2, i % 2)
        # Varsayılan birkaçı seçili gelsin
        self.checks["flip_h"].setChecked(True)
        self.checks["rotate"].setChecked(True)
        self.checks["brightness"].setChecked(True)
        aug_group.setLayout(aug_grid)
        layout.addWidget(aug_group)

        # Görsel başına üretilecek sayı
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Her görselden kaç yeni üretilsin:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 20)
        self.count_spin.setValue(3)
        count_layout.addWidget(self.count_spin)
        count_layout.addStretch()
        layout.addLayout(count_layout)

        # Başlat
        self.btn_start = QPushButton("Augmentation Başlat")
        self.btn_start.clicked.connect(self.start_aug)
        layout.addWidget(self.btn_start)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        layout.addWidget(self.log_area)

        self.setLayout(layout)

    def select_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Kaynak Klasörü Seç")
        if not folder:
            return
        self.src_line.setText(folder)

        self.images_dir, self.labels_dir = resolve_dataset_dirs(folder)

        self.log_area.append(f"Görsel klasörü: {self.images_dir}")
        self.log_area.append(f"Etiket klasörü: {self.labels_dir}")
        if not dir_has_images(self.images_dir):
            self.log_area.append(
                "UYARI: Seçilen klasörde doğrudan görsel yok. Lütfen görsellerin "
                "bulunduğu klasörü seçin (örn. .../images/train)."
            )
        elif not os.path.isdir(self.labels_dir):
            self.log_area.append(
                f"UYARI: Etiket klasörü bulunamadı: {self.labels_dir}"
            )

    def select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Çıktı Klasörü Seç")
        if folder:
            self.out_line.setText(folder)
            self.output_dir = folder

    def start_aug(self):
        if not self.images_dir or not self.output_dir:
            self.log_area.append("HATA: Kaynak ve çıktı klasörü seçin.")
            return

        config = {key: cb.isChecked() for key, cb in self.checks.items()}
        if not any(config.values()):
            self.log_area.append("HATA: En az bir augmentation seçin.")
            return

        # Etiket klasörü doğrulaması: bulunamazsa ya da hiç eşleşme yoksa durdur
        if not os.path.isdir(self.labels_dir):
            self.log_area.append(
                f"HATA: Etiket klasörü bulunamadı: {self.labels_dir}\n"
                "Kaynak olarak görsellerin klasörünü seçin (etiketler images/ → "
                "labels/ kuralıyla aranır)."
            )
            return
        image_files = [f for f in os.listdir(self.images_dir)
                       if f.lower().endswith(VALID_IMG_EXT)]
        matched = sum(
            1 for f in image_files
            if os.path.exists(os.path.join(self.labels_dir,
                                           os.path.splitext(f)[0] + ".txt"))
        )
        if matched == 0:
            self.log_area.append(
                f"HATA: {len(image_files)} görselin hiçbiriyle eşleşen etiket yok.\n"
                f"Etiket klasörü: {self.labels_dir}\nİşlem durduruldu."
            )
            return

        # Veri seti pose (keypoint) mu, detect mi? Etiketlerden otomatik algıla.
        num_kpts = keypoint_count_from_dir(self.labels_dir)
        flip_idx = []
        if num_kpts > 0:
            # flip_idx aktif projeden gelir; keypoint sayısıyla eşleşmeli
            if len(project_state.flip_idx) == num_kpts:
                flip_idx = list(project_state.flip_idx)

        self.btn_start.setEnabled(False)
        self.progress.setValue(0)
        self.log_area.clear()
        self.log_area.append("Augmentation başlatılıyor...")
        if num_kpts > 0:
            self.log_area.append(f"Keypoint verisi algılandı (K={num_kpts}).")
            if config.get("flip_h") and not flip_idx:
                self.log_area.append(
                    "UYARI: flip_idx tanımlı değil (Settings'ten doğru data.yaml'ı "
                    "seçin). Yatay çevirme, sol/sağ karışmasın diye atlanacak."
                )

        self.worker = AugmentationWorker(
            self.images_dir, self.labels_dir, self.output_dir,
            config, self.count_spin.value(),
            num_keypoints=num_kpts, flip_idx=flip_idx
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log_area.append)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, count):
        self.btn_start.setEnabled(True)
        self.log_area.append(f"\nİşlem bitti. Toplam {count} yeni görsel üretildi.")