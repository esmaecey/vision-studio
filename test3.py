from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n-pose.pt')
results = model('test_bend.jpg')

def calculate_angle(a, b, c):
    """a, b, c üç nokta (x,y). b köşe noktası (açının ölçüldüğü nokta)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return angle

CONF_THRESHOLD = 0.5  # bu değerin altındaki noktalara güvenmeyeceğiz

for result in results:
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        print("Kişi tespit edilemedi.")
        continue
    
    keypoints_xy = result.keypoints.xy[0]      # koordinatlar
    keypoints_conf = result.keypoints.conf[0]  # her noktanın güven skoru
    
    # Index tanımları (COCO formatı)
    idx = {
        "left_shoulder": 5, "right_shoulder": 6,
        "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14
    }
    
    def is_reliable(name):
        return keypoints_conf[idx[name]] >= CONF_THRESHOLD
    
    def point(name):
        return keypoints_xy[idx[name]].tolist()
    
    angles = []
    
    # Sol taraf güvenilirse hesapla
    if is_reliable("left_shoulder") and is_reliable("left_hip") and is_reliable("left_knee"):
        left_angle = calculate_angle(point("left_shoulder"), point("left_hip"), point("left_knee"))
        angles.append(left_angle)
        print(f"Sol taraf açısı: {left_angle:.2f} derece")
    else:
        print("Sol taraf noktaları yeterince net değil, hesaba katılmadı.")
    
    # Sağ taraf güvenilirse hesapla
    if is_reliable("right_shoulder") and is_reliable("right_hip") and is_reliable("right_knee"):
        right_angle = calculate_angle(point("right_shoulder"), point("right_hip"), point("right_knee"))
        angles.append(right_angle)
        print(f"Sağ taraf açısı: {right_angle:.2f} derece")
    else:
        print("Sağ taraf noktaları yeterince net değil, hesaba katılmadı.")
    
    # Sonuç
    if len(angles) == 0:
        print("⚠️ Yeterli güvenilir nokta yok, analiz yapılamadı.")
    else:
        final_angle = sum(angles) / len(angles)
        print(f"\nOrtalama Bel/Gövde Açısı: {final_angle:.2f} derece")
        
        if final_angle < 135:
            print("⚠️ TEHLİKELİ DURUŞ - Bel fazla bükülmüş!")
        else:
            print("✅ Güvenli duruş")