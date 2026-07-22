"""
Detect Labeling modülü için yardımcı fonksiyonlar.
YOLO formatında etiket okuma/yazma ve koordinat dönüşümleri.
"""
import os
import cv2
import numpy as np 

def yolo_to_pixel(x_center, y_center, w, h, img_width, img_height):
    """YOLO normalize koordinatlarını piksel koordinatlarına çevirir."""
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    w_px = w * img_width
    h_px = h * img_height
    x1 = x_center_px - w_px / 2
    y1 = y_center_px - h_px / 2
    return x1, y1, w_px, h_px


def pixel_to_yolo(x1, y1, w_px, h_px, img_width, img_height):
    """Piksel koordinatlarını YOLO normalize formatına çevirir."""
    x_center = (x1 + w_px / 2) / img_width
    y_center = (y1 + h_px / 2) / img_height
    w = w_px / img_width
    h = h_px / img_height
    return x_center, y_center, w, h


def load_yolo_labels(label_path):
    """
    Bir .txt dosyasından YOLO etiketlerini okur.

    Satırda 5'ten fazla token varsa (pose formatı: sınıf + kutu + Kx(x,y,v)),
    ilk 5 değer kutu olarak okunur; kalan keypoint token'ları 'tail' olarak
    korunur. Böylece kutu düzenlense bile keypoint verisi kaybolmaz; kayıtta
    geri yazılır.

    Dönüş: [(class_id, xc, yc, w, h, tail), ...]  (tail: str, yoksa "")
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x_center, y_center, w, h = map(float, parts[1:5])
            tail = " ".join(parts[5:])  # pose keypoint verisi (varsa)
            boxes.append((class_id, x_center, y_center, w, h, tail))
    return boxes


def save_yolo_labels(label_path, boxes):
    """
    boxes: [(class_id, xc, yc, w, h[, tail]), ...] listesini .txt dosyasına yazar.
    tail (pose keypoint verisi) varsa kutudan sonra korunarak yazılır.
    """
    with open(label_path, "w") as f:
        for box in boxes:
            class_id, x_center, y_center, w, h = box[:5]
            tail = box[5] if len(box) > 5 else ""
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            if tail:
                line += " " + tail
            f.write(line + "\n")


def get_class_color(class_id):
    """Her sınıf için sabit, birbirinden belirgin şekilde ayrılan bir renk döner."""
    palette = [
        (255, 0, 0),      # 0 boots - kırmızı
        (0, 200, 0),      # 1 gloves - yeşil
        (0, 100, 255),    # 2 goggles - mavi
        (255, 255, 0),    # 3 helmet - sarı
        (255, 0, 255),    # 4 no-boots - macenta
        (0, 255, 255),    # 5 no-gloves - camgöbeği
        (255, 128, 0),    # 6 no-goggles - turuncu
        (128, 0, 255),    # 7 no-helmet - mor
        (255, 255, 255),  # 8 no-vest - beyaz
        (0, 0, 0),        # 9 vest - siyah
    ]
    return palette[class_id % len(palette)]

import cv2
import numpy as np


def draw_boxes_on_image(image_path, boxes, class_names):
    """Görselin üzerine kutuları çizip (vis için) BGR numpy array döner.
    Türkçe karakterli yolları da destekler."""
    # cv2.imread Türkçe yolda çalışmaz; np.fromfile ile okuyoruz
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]

    for box in boxes:
        class_id, xc, yc, bw, bh = box[:5]  # tail (pose) varsa yok say
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        r, g, b = get_class_color(class_id)
        color = (b, g, r)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = class_names[class_id] if class_id < len(class_names) else str(class_id)
        cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def save_image_unicode(save_path, img):
    """cv2.imwrite'ın Türkçe yol sorununu aşan güvenli kaydetme."""
    ext = os.path.splitext(save_path)[1]
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(save_path)
        return True
    return False

    for class_id, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        r, g, b = get_class_color(class_id)
        color = (b, g, r)  # OpenCV BGR kullanır

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = class_names[class_id] if class_id < len(class_names) else str(class_id)
        cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img