"""
Keypoint Labeling yardımcıları — COCO 17 keypoint şablonu, iskelet bağlantıları,
YOLO-Pose format okuma/yazma, keypoint renk paleti.
"""
import os
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Renk paleti — her keypoint index'i kendine ait sabit bir renkte gösterilsin.
# 21 noktalı el (MediaPipe/Ultralytics sırası) için parmak bazlı gruplama;
# diğer şablonlar (COCO 17, özel) için ayırt edilebilir genel palet.
# Renkler RGB döner; index'e göre STABİL'dir (kpt_shape/flip_idx ile tutarlı).
# ---------------------------------------------------------------------------

# Standart 21 noktalı el: parmak grupları (index listeleri)
_HAND_FINGERS = [
    ("thumb",  [1, 2, 3, 4]),
    ("index",  [5, 6, 7, 8]),
    ("middle", [9, 10, 11, 12]),
    ("ring",   [13, 14, 15, 16]),
    ("pinky",  [17, 18, 19, 20]),
]
# Her parmak için taban (koyu) renk — uca doğru açılır (lighten)
_FINGER_BASE = {
    "thumb":  (220, 40, 40),    # kırmızı
    "index":  (240, 150, 20),   # turuncu
    "middle": (40, 180, 70),    # yeşil
    "ring":   (40, 130, 245),   # mavi
    "pinky":  (180, 70, 230),   # mor
}
_WRIST_COLOR = (245, 245, 245)  # bilek (0) — açık gri/beyaz

# Genel ayırt edilebilir palet (Sasha Trubetskoy'un 20 renk seti + ekler)
_DISTINCT = [
    (230, 25, 75), (60, 180, 75), (255, 200, 20), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (170, 220, 40), (250, 150, 190),
    (0, 160, 160), (200, 160, 255), (170, 110, 40), (255, 220, 130), (170, 0, 40),
    (170, 255, 195), (150, 150, 0), (255, 160, 110), (60, 100, 220), (128, 128, 128),
    (255, 100, 180), (30, 200, 130), (120, 60, 220), (210, 90, 40),
]


def _lighten(rgb, t):
    """rgb rengini beyaza doğru t (0..1) oranında açar."""
    return tuple(int(c + (255 - c) * t) for c in rgb)


def _finger_of(index):
    """21'lik el düzeninde index hangi parmakta? (bilek/geçersizse None)"""
    for name, idxs in _HAND_FINGERS:
        if index in idxs:
            return name, idxs
    return None


def keypoint_color(index, total):
    """
    Bir keypoint index'i için sabit RGB renk döner.
    total == 21 ise parmak bazlı renklendirme; aksi halde genel palet.
    """
    if total == 21:
        if index == 0:
            return _WRIST_COLOR
        fg = _finger_of(index)
        if fg is not None:
            name, idxs = fg
            pos = idxs.index(index)            # 0 (taban) .. 3 (uç)
            return _lighten(_FINGER_BASE[name], 0.18 * pos)
    return _DISTINCT[index % len(_DISTINCT)]


def skeleton_edge_color(a, b, total):
    """Bir iskelet kenarı (a-b) için RGB renk. El ise parmak grubuna göre."""
    if total == 21:
        fa = _finger_of(a)
        fb = _finger_of(b)
        name = (fa or fb)[0] if (fa or fb) else None
        if name is not None:
            return _FINGER_BASE[name]
        return (0, 200, 255)  # bileğe bağlanan / gruplanamayan kenarlar
    # Genel: kenarı uç (b) noktasının rengiyle boya
    return keypoint_color(b, total)




def save_pose_label(label_path, persons, img_width, img_height):
    """
    persons: her biri {'bbox': (x1,y1,x2,y2), 'keypoints': [(x,y,v), ...17]} olan liste.
    YOLO-Pose formatında kaydeder:
    class cx cy w h  px1 py1 v1  px2 py2 v2 ... (normalize)
    """
    lines = []
    for person in persons:
        x1, y1, x2, y2 = person['bbox']
        cx = ((x1 + x2) / 2) / img_width
        cy = ((y1 + y2) / 2) / img_height
        w = abs(x2 - x1) / img_width
        h = abs(y2 - y1) / img_height

        parts = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]
        for (px, py, v) in person['keypoints']:
            parts.append(f"{px / img_width:.6f} {py / img_height:.6f} {v}")
        lines.append(" ".join(parts))

    with open(label_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_pose_label(label_path, img_width, img_height):
    """YOLO-Pose txt dosyasını okuyup persons listesi döner (piksel koordinatlı)."""
    persons = []
    if not os.path.exists(label_path):
        return persons

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w / 2) * img_width
            y1 = (cy - h / 2) * img_height
            x2 = (cx + w / 2) * img_width
            y2 = (cy + h / 2) * img_height

            keypoints = []
            kp_data = parts[5:]
            for i in range(0, len(kp_data), 3):
                if i + 2 < len(kp_data):
                    px = float(kp_data[i]) * img_width
                    py = float(kp_data[i + 1]) * img_height
                    v = int(float(kp_data[i + 2]))
                    keypoints.append((px, py, v))

            persons.append({'bbox': (x1, y1, x2, y2), 'keypoints': keypoints})
    return persons

def draw_pose_on_image(image_path, persons, skeleton):
    """Görselin üzerine iskelet + noktaları çizip BGR array döner."""
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    for person in persons:
        kps = person['keypoints']
        total = len(kps)

        # İskelet kenarları — parmak grubuna göre renkli (RGB -> BGR)
        for (a, b) in skeleton:
            if a < len(kps) and b < len(kps):
                xa, ya, va = kps[a]
                xb, yb, vb = kps[b]
                if va > 0 and vb > 0:
                    r, g, bl = skeleton_edge_color(a, b, total)
                    cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (bl, g, r), 2)

        # Noktalar — her index kendi rengiyle (RGB -> BGR); v==1 içi boş
        for i, (x, y, v) in enumerate(kps):
            if v > 0:
                r, g, bl = keypoint_color(i, total)
                bgr = (bl, g, r)
                if v == 2:
                    cv2.circle(img, (int(x), int(y)), 4, bgr, -1)
                else:
                    cv2.circle(img, (int(x), int(y)), 4, bgr, 1)

    return img


def save_image_unicode(save_path, img):
    """cv2.imwrite'ın Türkçe yol sorununu aşan güvenli kaydetme."""
    ext = os.path.splitext(save_path)[1]
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(save_path)
        return True
    return False