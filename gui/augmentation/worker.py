"""
Augmentation worker — seçilen dönüşümleri veri setine uygular, arka planda çalışır.
Detect (bounding box) ve Pose (keypoint) etiketlerini otomatik dönüştürür.

Pose desteği:
  - Etiketler (class + box + K×(x,y,v)) doğru okunup yazılır.
  - Yatay çevirme MANUEL yapılır: x aynalanır ve flip_idx ile sol/sağ keypoint
    indeksleri takas edilir (albumentations bunu tek başına yapmaz).
  - Döndürme/dikey çevirme vb. albumentations keypoint_params ile keypoint'leri
    de dönüştürür. Kutu, keypoint'lere köşe noktaları eklenip birlikte
    dönüştürülerek yeniden hesaplanır (instance hizalaması garanti).
  - Kadraj dışına çıkan keypoint'lerin görünürlüğü (v) 0 yapılır.
"""
import os
import copy
import random
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
    return ok


# ---------------- Detect (box) etiket okuma/yazma ----------------
def load_yolo_boxes(label_path):
    boxes, classes = [], []
    if not os.path.exists(label_path):
        return boxes, classes
    with open(label_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 5:
                classes.append(int(float(p[0])))
                boxes.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return boxes, classes


def save_yolo_boxes(label_path, boxes, classes):
    with open(label_path, "w") as f:
        for cls, box in zip(classes, boxes):
            f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")


# ---------------- Pose (keypoint) etiket okuma/yazma ----------------
def load_yolo_pose(label_path, K):
    """Pose etiketini okur. Dönüş: instances = [{cls, box:[cx,cy,w,h], kpts:[[x,y,v]*K]}]."""
    instances = []
    if not os.path.exists(label_path):
        return instances
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) != 5 + 3 * K:
                continue
            cls = int(float(p[0]))
            box = [float(x) for x in p[1:5]]
            vals = p[5:]
            kpts = [[float(vals[3 * j]), float(vals[3 * j + 1]), float(vals[3 * j + 2])]
                    for j in range(K)]
            instances.append({"cls": cls, "box": box, "kpts": kpts})
    return instances


def save_yolo_pose(label_path, instances):
    with open(label_path, "w") as f:
        for ins in instances:
            box = ins["box"]
            parts = [str(ins["cls"]), f"{box[0]:.6f}", f"{box[1]:.6f}",
                     f"{box[2]:.6f}", f"{box[3]:.6f}"]
            for (x, y, v) in ins["kpts"]:
                parts.append(f"{x:.6f} {y:.6f} {int(round(v))}")
            f.write(" ".join(parts) + "\n")


class AugmentationWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, images_dir, labels_dir, output_dir, transforms_config,
                 num_per_image, num_keypoints=0, flip_idx=None):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.transforms_config = transforms_config
        self.num_per_image = num_per_image
        self.num_keypoints = num_keypoints          # >0 ise pose modu
        self.flip_idx = list(flip_idx) if flip_idx else []

    # ---------------- Transform kurucular ----------------
    def _build_detect_transform(self):
        c = self.transforms_config
        augs = self._pixel_and_geo_augs(include_hflip=True)
        return A.Compose(
            augs,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
        )

    def _build_pose_transform(self):
        # Yatay çevirme pose'da MANUEL yapıldığından buraya dahil edilmez.
        augs = self._pixel_and_geo_augs(include_hflip=False)
        return A.Compose(
            augs,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def _pixel_and_geo_augs(self, include_hflip):
        c = self.transforms_config
        augs = []
        if include_hflip and c.get("flip_h"):
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
        return augs

    # ---------------- Ana döngü ----------------
    def run(self):
        try:
            out_images = os.path.join(self.output_dir, "images")
            out_labels = os.path.join(self.output_dir, "labels")
            os.makedirs(out_images, exist_ok=True)
            os.makedirs(out_labels, exist_ok=True)

            is_pose = self.num_keypoints > 0
            if is_pose:
                self.log.emit(f"Pose modu: {self.num_keypoints} keypoint algılandı.")
                if self.transforms_config.get("flip_h") and not self._valid_flip_idx():
                    self.log.emit(
                        "UYARI: Yatay çevirme seçili ama geçerli flip_idx yok "
                        "(data.yaml). Sol/sağ keypoint bozulmasın diye yatay "
                        "çevirme bu veri seti için ATLANIYOR."
                    )
                transform = self._build_pose_transform()
            else:
                transform = self._build_detect_transform()

            valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
            images = [f for f in os.listdir(self.images_dir) if f.lower().endswith(valid_ext)]
            if not images:
                self.log.emit(f"HATA: Görsel bulunamadı: {self.images_dir}")
                self.finished_signal.emit(0)
                return

            # --- Etiket ön kontrolü: kaç görselde etiket var, kaç satır okundu? ---
            self.log.emit(f"Görsel klasörü: {self.images_dir}")
            self.log.emit(f"Etiket klasörü: {self.labels_dir}")
            if not self.labels_dir or not os.path.isdir(self.labels_dir):
                self.log.emit(f"HATA: Etiket klasörü bulunamadı: {self.labels_dir}\n"
                              "Kaynak olarak görsellerin bulunduğu klasörü seçin; "
                              "etiketler standart YOLO yapısında (images/ → labels/) aranır.")
                self.finished_signal.emit(0)
                return

            labeled_images, total_lines = 0, 0
            for img_file in images:
                base = os.path.splitext(img_file)[0]
                lp = os.path.join(self.labels_dir, base + ".txt")
                if os.path.exists(lp):
                    with open(lp) as f:
                        lines = [ln for ln in f if ln.strip()]
                    if lines:
                        labeled_images += 1
                        total_lines += len(lines)
            self.log.emit(
                f"{labeled_images}/{len(images)} görselde etiket bulundu, "
                f"toplam {total_lines} satır okundu."
            )
            if total_lines == 0:
                self.log.emit(
                    "HATA: Hiç etiket satırı okunamadı — işlem durduruldu.\n"
                    f"Etiket klasörünü kontrol edin: {self.labels_dir}"
                )
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

                for i in range(self.num_per_image):
                    out_name = f"{base}_aug{i}"
                    try:
                        if is_pose:
                            ok = self._process_pose(transform, image, label_path,
                                                    out_images, out_labels, out_name)
                        else:
                            ok = self._process_detect(transform, image, label_path,
                                                      out_images, out_labels, out_name)
                        if ok:
                            generated += 1
                    except Exception as e:
                        self.log.emit(f"Uyarı ({img_file}, {i}): {str(e)}")

                self.progress.emit(int((idx + 1) / total * 100))

            self.log.emit(f"Tamamlandı! {generated} yeni görsel üretildi.")
            self.finished_signal.emit(generated)

        except Exception as e:
            self.log.emit(f"HATA: {str(e)}")
            self.finished_signal.emit(0)

    def _valid_flip_idx(self):
        return (len(self.flip_idx) == self.num_keypoints
                and sorted(self.flip_idx) == list(range(self.num_keypoints)))

    # ---------------- Detect işleme ----------------
    def _process_detect(self, transform, image, label_path, out_images, out_labels, out_name):
        boxes, classes = load_yolo_boxes(label_path)
        if boxes:
            result = transform(image=image, bboxes=boxes, class_labels=classes)
            aug_boxes, aug_classes = result["bboxes"], result["class_labels"]
        else:
            result = transform(image=image, bboxes=[], class_labels=[])
            aug_boxes, aug_classes = [], []
        aug_image = result["image"]
        imwrite_unicode(os.path.join(out_images, out_name + ".jpg"), aug_image)
        save_yolo_boxes(os.path.join(out_labels, out_name + ".txt"), aug_boxes, aug_classes)
        return True

    # ---------------- Pose işleme ----------------
    def _process_pose(self, transform, image, label_path, out_images, out_labels, out_name):
        K = self.num_keypoints
        H, W = image.shape[:2]
        instances = load_yolo_pose(label_path, K)

        img_i = image
        inst_i = copy.deepcopy(instances)

        # 1) Manuel yatay çevirme (x aynalama + flip_idx takası)
        do_flip = (self.transforms_config.get("flip_h")
                   and self._valid_flip_idx()
                   and random.random() < 0.5)
        if do_flip:
            img_i = cv2.flip(img_i, 1)
            for ins in inst_i:
                ins["box"][0] = 1.0 - ins["box"][0]  # cx aynala
                new_kpts = []
                for j in range(K):
                    sx, sy, sv = ins["kpts"][self.flip_idx[j]]
                    if sv > 0:
                        sx = 1.0 - sx  # sadece görünür noktayı aynala
                    new_kpts.append([sx, sy, sv])
                ins["kpts"] = new_kpts

        # 2) Keypoint listesini kur: her instance için K nokta + 4 kutu köşesi
        #    (köşeler de keypoint gibi dönüştürülüp kutu yeniden hesaplanır)
        points = []
        for ins in inst_i:
            for (x, y, v) in ins["kpts"]:
                points.append((x * W, y * H))
            cx, cy, bw, bh = ins["box"]
            x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
            x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
            points.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        result = transform(image=img_i, keypoints=points)
        aug_image = result["image"]
        aug_pts = result["keypoints"]
        Hn, Wn = aug_image.shape[:2]

        # 3) Instance'ları yeniden kur (her biri K+4 nokta)
        stride = K + 4
        out_instances = []
        for gi, ins in enumerate(inst_i):
            gbase = gi * stride
            real = aug_pts[gbase:gbase + K]
            corners = aug_pts[gbase + K:gbase + stride]

            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            bx1 = max(0.0, min(xs)); bx2 = min(float(Wn), max(xs))
            by1 = max(0.0, min(ys)); by2 = min(float(Hn), max(ys))
            if bx2 - bx1 < 1 or by2 - by1 < 1:
                continue  # kutu kadraj dışı → bu instance'ı yaz(ma)

            box = [((bx1 + bx2) / 2) / Wn, ((by1 + by2) / 2) / Hn,
                   (bx2 - bx1) / Wn, (by2 - by1) / Hn]

            kpts_out = []
            for j in range(K):
                x, y = real[j][0], real[j][1]
                v = inst_i[gi]["kpts"][j][2]
                inside = (0 <= x < Wn) and (0 <= y < Hn)
                if v > 0 and inside:
                    kpts_out.append((x / Wn, y / Hn, v))
                else:
                    # kadraj dışı veya zaten görünmez → v=0
                    cxp = min(max(x, 0.0), Wn - 1) / Wn
                    cyp = min(max(y, 0.0), Hn - 1) / Hn
                    kpts_out.append((cxp, cyp, 0))
            out_instances.append({"cls": ins["cls"], "box": box, "kpts": kpts_out})

        imwrite_unicode(os.path.join(out_images, out_name + ".jpg"), aug_image)
        save_yolo_pose(os.path.join(out_labels, out_name + ".txt"), out_instances)
        return True
