"""
Keypoint Labeling yardımcıları — COCO 17 keypoint şablonu, iskelet bağlantıları,
YOLO-Pose format okuma/yazma.
"""
import os
import cv2
import numpy as np




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

        for (a, b) in skeleton:
            if a < len(kps) and b < len(kps):
                xa, ya, va = kps[a]
                xb, yb, vb = kps[b]
                if va > 0 and vb > 0:
                    cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255, 200, 0), 2)

        for (x, y, v) in kps:
            if v > 0:
                color = (0, 0, 255) if v == 2 else (0, 165, 255)
                cv2.circle(img, (int(x), int(y)), 4, color, -1)

    return img


def save_image_unicode(save_path, img):
    """cv2.imwrite'ın Türkçe yol sorununu aşan güvenli kaydetme."""
    ext = os.path.splitext(save_path)[1]
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(save_path)
        return True
    return False