"""
Augmentation worker — seçilen dönüşümleri veri setine uygular, arka planda çalışır.
Bounding box etiketlerini de otomatik dönüştürür (albumentations).
"""
import os
import cv2
import numpy as np
import albumentations as A
from PyQt5.QtCore import QThread, pyqtSignal


def imread_unicode(path):
    arr = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    ok, enc = cv2.imencode(ext, img)
    if ok:
        enc.tofile(path)


def load_yolo_boxes(label_path):
    boxes, classes = [], []
    if not os.path.exists(label_path):
        return boxes, classes
    with open(label_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 5:
                classes.append(int(p[0]))
                boxes.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return boxes, classes


def save_yolo_boxes(label_path, boxes, classes):
    with open(label_path, "w") as f:
        for cls, box in zip(classes, boxes):
            f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")


class AugmentationWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, images_dir, labels_dir, output_dir, transforms_config, num_per_image):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.transforms_config = transforms_config
        self.num_per_image = num_per_image

    def _build_transform(self):
        augs = []
        c = self.transforms_config
        if c.get("flip_h"):
            augs.append(A.HorizontalFlip(p=0.5))
        if c.get("flip_v"):
            augs.append(A.VerticalFlip(p=0.5))
        if c.get("rotate"):
            augs.append(A.Rotate(limit=20, p=0.7))
        if c.get("brightness"):
            augs.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6))
        if c.get("hue"):
            augs.append(A.HueSaturationValue(p=0.5))
        if c.get("blur"):
            augs.append(A.Blur(blur_limit=3, p=0.4))
        if c.get("noise"):
            augs.append(A.GaussNoise(p=0.4))
        if c.get("clahe"):
            augs.append(A.CLAHE(p=0.4))

        return A.Compose(
            augs,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3)
        )

    def run(self):
        try:
            out_images = os.path.join(self.output_dir, "images")
            out_labels = os.path.join(self.output_dir, "labels")
            os.makedirs(out_images, exist_ok=True)
            os.makedirs(out_labels, exist_ok=True)

            transform = self._build_transform()

            valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
            images = [f for f in os.listdir(self.images_dir) if f.lower().endswith(valid_ext)]

            if not images:
                self.log.emit("HATA: Görsel bulunamadı.")
                self.finished_signal.emit(0)
                return

            total = len(images)
            generated = 0

            for idx, img_file in enumerate(images):
                img_path = os.path.join(self.images_dir, img_file)
                base = os.path.splitext(img_file)[0]
                label_path = os.path.join(self.labels_dir, base + ".txt")

                image = imread_unicode(img_path)
                if image is None:
                    continue
                boxes, classes = load_yolo_boxes(label_path)

                for i in range(self.num_per_image):
                    try:
                        if boxes:
                            result = transform(image=image, bboxes=boxes, class_labels=classes)
                            aug_boxes = result["bboxes"]
                            aug_classes = result["class_labels"]
                        else:
                            result = transform(image=image, bboxes=[], class_labels=[])
                            aug_boxes, aug_classes = [], []

                        aug_image = result["image"]

                        out_name = f"{base}_aug{i}"
                        imwrite_unicode(os.path.join(out_images, out_name + ".jpg"), aug_image)
                        save_yolo_boxes(os.path.join(out_labels, out_name + ".txt"), aug_boxes, aug_classes)
                        generated += 1
                    except Exception as e:
                        self.log.emit(f"Uyarı ({img_file}, {i}): {str(e)}")

                self.progress.emit(int((idx + 1) / total * 100))

            self.log.emit(f"Tamamlandı! {generated} yeni görsel üretildi.")
            self.finished_signal.emit(generated)

        except Exception as e:
            self.log.emit(f"HATA: {str(e)}")
            self.finished_signal.emit(0)