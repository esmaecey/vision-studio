from ultralytics import YOLO
import cv2

model = YOLO('yolov8n-pose.pt')

# Fotoğraf üzerinde tahmin yap
results = model('test_bend.jpg')

# Sonucu al
for result in results:
    keypoints = result.keypoints  # eklem noktaları
    print("Tespit edilen kişi sayısı:", len(keypoints))
    print("Eklem koordinatları:\n", keypoints.xy)  # (x, y) piksel koordinatları
    
    # Görselleştirilmiş halini kaydet
    result.save(filename='sonuc.jpg')

print("İşlem tamam! sonuc.jpg dosyasına bak.")